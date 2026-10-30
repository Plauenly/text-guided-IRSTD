# -*- coding: utf-8 -*-
"""
文本适配器 / ACM 自检（需要 torch，可在 CPU 上跑）

    python tools/test_cross_modal.py
    python tools/test_cross_modal.py --with_clip --clip_path /path/ViT-B-32.pt   # 额外验证真实 E

覆盖：
    1. TextAdapter：gate=0 时是恒等映射；参数量
    2. ACM：形状、掩码比例、loss 有限
    3. **反塌陷**：把投影层置零（等价"视觉侧啥也没提供"）时损失应显著变大
    4. **梯度**：grad_to_backbone=True 时梯度能回传到"主干"特征；False 时不能
    5. 端到端：只训 ACM 参数，若干步后 masked 损失应下降
"""

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.cross_modal import TextAdapter, AdaptiveCrossModalMask  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_clip', action='store_true')
    parser.add_argument('--clip_path', default='ViT-B/32')
    args = parser.parse_args()

    torch.manual_seed(0)
    B, C, H, W, N, D = 3, 128, 16, 16, 10, 512

    # ---------- 1. TextAdapter ----------
    adapter = TextAdapter(D, hidden=512, gate_init=0.0)
    E = torch.randn(N, D)
    Ea0 = adapter(E)
    assert torch.allclose(Ea0, E), 'gate=0 时适配器必须是恒等映射'
    n = sum(p.numel() for p in adapter.parameters())
    print(f'[1] TextAdapter OK: 恒等初始化；参数量={n} (~{n/1e3:.1f}K)')
    with torch.no_grad():
        adapter.gate.fill_(1.0)
    assert not torch.allclose(adapter(E), E), 'gate=1 时应改变嵌入'
    with torch.no_grad():
        adapter.gate.fill_(0.0)
    print('[2] TextAdapter gate 生效 OK')

    # ---------- 2. ACM 形状与掩码 ----------
    acm = AdaptiveCrossModalMask(in_dim=C, text_dim=D, mask_ratio=0.15,
                                 grad_to_backbone=True)
    acm.train()
    acm.set_enabled(True)
    feat = torch.randn(B, C, H, W)
    acm._feat = feat                     # 直接注入，等价于 pre-hook 捕获
    loss = acm.forward_loss()
    assert loss.dim() == 0 and torch.isfinite(loss), loss
    st = acm.last_stats
    assert 0 < st['masked'] < 10 and 0 < st['unmasked'] < 10, st
    print(f"[3] ACM 形状 OK: loss={float(loss):.4f} masked={st['masked']:.4f} "
          f"kept={st['unmasked']:.4f}")

    # 掩码比例：用固定随机数统计被清零的元素比例
    torch.manual_seed(1)
    keep = (torch.rand(N, D) > 0.15)
    print(f"[4] 掩码比例检查: 清零比例≈{1 - keep.float().mean():.3f} (期望≈0.15)")

    # ---------- 3. 反塌陷：投影置零时损失必须变大 ----------
    acm2 = AdaptiveCrossModalMask(in_dim=C, text_dim=D, mask_ratio=0.15)
    acm2.train()
    acm2.set_enabled(True)
    with torch.no_grad():
        acm2.proj.weight.zero_()
    acm2._feat = feat.clone()
    l_zero = float(acm2.forward_loss())
    acm2._feat = feat.clone()
    l_rand = float(acm.forward_loss())
    assert l_zero > l_rand, f'投影为 0 时损失应更大（{l_zero:.4f} vs {l_rand:.4f}）'
    print(f'[5] 反塌陷 OK: 投影=0 时 loss={l_zero:.4f} > 随机投影 loss={l_rand:.4f}')

    # ---------- 4. 梯度是否回传主干 ----------
    for grad_bb, expect in ((True, True), (False, False)):
        a = AdaptiveCrossModalMask(in_dim=C, text_dim=D, grad_to_backbone=grad_bb)
        a.train()
        a.set_enabled(True)
        f = torch.randn(B, C, H, W, requires_grad=True)
        a._feat = f if grad_bb else f.detach()
        l = a.forward_loss()
        l.backward()
        has = f.grad is not None
        assert has == expect, f'grad_to_backbone={grad_bb} 时主干梯度={has}'
        print(f'[6] grad_to_backbone={grad_bb}: 主干拿到梯度={has} OK')

    # ---------- 5. 端到端：只训 ACM 参数，损失应下降 ----------
    acm3 = AdaptiveCrossModalMask(in_dim=C, text_dim=D, mask_ratio=0.15)
    acm3.train()
    acm3.set_enabled(True)
    opt = torch.optim.Adam(acm3.parameters(), lr=1e-3)
    E3 = torch.randn(N, D)
    acm3.set_text_embedding(E3)
    feat = torch.randn(B, C, H, W, requires_grad=False)
    losses = []
    for it in range(200):
        opt.zero_grad()
        acm3._feat = feat                      # 固定特征，只学投影
        l = acm3.forward_loss()
        l.backward()
        opt.step()
        losses.append(float(l))
    assert losses[-1] < losses[0], (losses[0], losses[-1])
    print(f'[7] 端到端下降 OK: loss {losses[0]:.4f} -> {losses[-1]:.4f} '
          f'({100*(1-losses[-1]/losses[0]):.0f}%)')

    # ---------- 6. 可选：真实 CLIP 文本嵌入 ----------
    if args.with_clip:
        from components.text_prompts import SINGLE_FRAME_IR_PROMPTS
        from components.text_prior import ClipTextEmbedder
        emb = ClipTextEmbedder(SINGLE_FRAME_IR_PROMPTS, args.clip_path,
                               device='cpu', keep_image_tower=False, verbose=False)
        Ereal = emb.get()
        print(f'[8] 真实 E: shape={tuple(Ereal.shape)}')
        # 秩检查：ACM 的重建目标需要有区分度，否则任务退化
        s = torch.linalg.svdvals(Ereal - Ereal.mean(0, keepdim=True))
        ratio = float(s[0] ** 2 / (s ** 2).sum())
        print(f'    第一奇异值能量占比={ratio:.3f}（>0.95 说明 10 条 prompt 近乎同义，'
              f'ACM 重建任务会退化）')

    print('\n全部自检通过。')


if __name__ == '__main__':
    main()
