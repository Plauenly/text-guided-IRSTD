# -*- coding: utf-8 -*-
"""
文本先验模块自检脚本（不依赖数据集，纯随机张量）。

用法：
    cd /workspace/PAL+TATE/PAL-main
    python tools/test_text_prior.py                 # 只测模块与 MSDA 接入（不需要 CLIP 权重）
    python tools/test_text_prior.py --with_clip     # 额外测试 CLIP 文本嵌入（需要 clip 包 + 权重）
"""

import os
import sys
import argparse

# 把 PAL-main 根目录加入 sys.path，保证能 import components / model
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from components.text_prior import (
    TextPriorAttention,
    attach_text_prior,
    MODEL_TEXT_PRIOR_CFG,
)


def test_attention_module():
    print('\n[Test 1] TextPriorAttention 单模块 ------------------------------------')
    feat_dim, text_dim, N = 64, 512, 10
    m = TextPriorAttention(feat_dim=feat_dim, text_dim=text_dim, use_gate=True, gate_init=0.0)
    m.set_text_embedding(torch.randn(N, text_dim))
    m.eval()

    x = torch.randn(2, feat_dim, 16, 16)
    with torch.no_grad():
        y = m(x)

    assert y.shape == x.shape, f'输出形状应为 {tuple(x.shape)}，实际 {tuple(y.shape)}'
    print(f'  输入 {tuple(x.shape)} -> 输出 {tuple(y.shape)}  OK')
    print(f'  注意力图 _last_attention: {tuple(m._last_attention.shape)}  OK')

    # gate_init=0 时，初始化应严格等价于恒等映射
    assert torch.allclose(y, x, atol=1e-6), 'gate_init=0 时输出应等于输入'
    print('  gate_init=0 -> 恒等映射校验通过  OK')

    # 门控可学习
    assert m.alpha.requires_grad, 'alpha 应可训练'
    print('  alpha 可训练  OK')

    # 注意力图空间维应归一化（和为 1）
    s = m._last_attention.flatten(2).sum(dim=-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5), '空间 softmax 未归一化'
    print('  空间 softmax 归一化校验通过  OK')


def test_msda_integration():
    print('\n[Test 2] MSDA + forward hook 接入 -------------------------------------')
    from model.MSDA.MSDA_no_sigmoid import MSDA_No_Sigmoid

    model = MSDA_No_Sigmoid()
    n_before = sum(p.numel() for p in model.parameters())

    attach_text_prior(model, torch.randn(10, 512), choose_model='MSDA', verbose=True)

    n_after = sum(p.numel() for p in model.parameters())
    print(f'  参数量: {n_before:,} -> {n_after:,}  (+{n_after - n_before:,})')

    # 文本模块的参数确实进入了 model.parameters()（优化器才会更新它）
    text_ids = {id(p) for p in model.text_prior.parameters()}
    model_ids = {id(p) for p in model.parameters()}
    assert text_ids.issubset(model_ids), 'text_prior 参数未包含在 model.parameters() 中'
    print('  text_prior 参数已被 model.parameters() 覆盖  OK')

    # 前向：hook 能拿到 64ch bottleneck 并返回同形状特征
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 256, 256))
    print(f'  前向输出: {tuple(out.shape)}  OK')
    assert out.shape[0] == 1 and out.shape[1] == 1
    print(f'  注入层捕捉到的特征图: {tuple(model.text_prior._last_attention.shape)}  OK')

    # 反向：梯度能流到文本模块
    model.train()
    out = model(torch.randn(1, 3, 256, 256))
    loss = out.mean()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.text_prior.parameters())
    assert has_grad, '梯度未流入文本先验模块'
    print('  反向传播梯度已到达文本先验模块  OK')


def test_clip_embedder(clip_model_or_path='ViT-B/32'):
    print('\n[Test 3] ClipTextEmbedder ---------------------------------------------')
    from components.text_prior import ClipTextEmbedder
    from components.text_prompts import SINGLE_FRAME_IR_PROMPTS

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    emb = ClipTextEmbedder(SINGLE_FRAME_IR_PROMPTS, model_name_or_path=clip_model_or_path, device=device)
    e = emb.get()
    print(f'  文本嵌入: {tuple(e.shape)}  dtype={e.dtype}  OK')
    assert e.dim() == 2 and e.size(0) == len(SINGLE_FRAME_IR_PROMPTS) and e.size(1) == 512


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_clip', action='store_true',
                        help='是否测试 CLIP 文本嵌入（需要 clip 包与权重）')
    parser.add_argument('--clip_path', default='ViT-B/32',
                        help="CLIP 模型名（如 ViT-B/32，自动下载）或本地 .pt 绝对路径，"
                             "例如 /mnt/xxx/ViT-B-32.pt")
    args = parser.parse_args()

    print('已在 MODEL_TEXT_PRIOR_CFG 中配置的模型:', list(MODEL_TEXT_PRIOR_CFG.keys()))

    test_attention_module()
    test_msda_integration()
    if args.with_clip:
        test_clip_embedder(args.clip_path)

    print('\n全部测试通过 ✓')


if __name__ == '__main__':
    main()
