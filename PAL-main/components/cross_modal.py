# -*- coding: utf-8 -*-
"""
文本适配器 + 自适应跨模态掩码（Adaptive Cross-modal Masking, ACM）
=================================================================
来源与改造
----------
SeViL (AAAI'26) 的 ACM 是"文本特征/图像特征各自随机掩码 → 用对方恢复 → MSE"。
本文件把它改造成适配 PAL+MSDA 的形式，并修掉了原式在本设定下的三处问题：

1) **维度**：MSDA 融合前的 bottleneck 是 128ch×16×16，CLIP 文本是 512 维。
   因此把视觉侧投影到 **512 维**（`proj: 1x1 conv 128→512`），文本侧保持 512 维，
   重建目标就是**真正的 CLIP 文本嵌入**（不做可学习的 target 投影，避免目标被
   投影层吞掉语义）。对应"融合前"的张量 = `stage.layer5_4` 的 **输入**。

2) **旁路塌陷**：若写成 `out = alpha*attn·V + text_mask` 再对 out 求 MSE，
   网络只要把 alpha 学到 0 就退化成"用适配器猜被 mask 的 token"的纯文本任务，
   视觉侧拿不到梯度。本实现 **只对 `attn·V` 求损失**，且输入 query 是被掩码后的
   文本嵌入（被 mask 的位置置 0 后重新 L2 归一化），因此想复现被 mask 的语义，
   只能从视觉特征里读 —— 梯度必然回传到 MSDA（除非显式 detach）。

3) **数值**：Q、V 都做 L2 归一化，注意力用 `softmax(Q·Kᵀ/√d)`；重建目标也归一化，
   于是 MSE 等价于 2(1-cos)，尺度稳定。嵌入按元素级随机掩码（默认 15%），
   不做"整行置零"。

数据流
------
    E   = clip_text(prompts)                  [N,512]   （缓存，可被 TextAdapter 适配）
    En  = E/||E||                             [N,512]   ← 重建目标
    keep= rand > mask_ratio                   [N,512]   1=保留
    Q   = normalize(En*keep)                  [N,512]   ← 被 mask 的位置置 0 后再归一化
    V   = normalize(proj(prefusion))          [B,512,256]（16*16=256 个空间位置，K 与 V 共用）
    attn= softmax(Q·Vᵀ/√512)                  [B,N,256]
    out = attn·V                              [B,N,512]
    loss= MSE(out, En)[被 mask 的元素]         ← 视觉特征必须"读出"文本语义

使用方式（train_model.py）
-------------------------
    adapter = TextAdapter(512, gate_init=0.0)          # Ea = E + gate*MLP(E)，零初始化
    model.add_module('text_adapter', adapter)          # 只注册一次，进 model.parameters()
    attach_text_prior(model, E, ..., text_adapter=adapter)
    acm = attach_cross_modal(model, E, adapter, choose_model='MSDA')
    ...
    l_acm = acm.forward_loss()                         # 每个训练 step 调用一次
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. 文本适配器：在缓存的 CLIP 文本嵌入上做轻量微调
# =============================================================================
class TextAdapter(nn.Module):
    """Ea = E + gate * MLP(E)。

    `gate` 初值为 0 ⇒ 训练开始时该模块是**恒等映射**，与"不接适配器"完全等价；
    之后由梯度决定要不要用适配后的文本。这比直接微调 CLIP 文本塔便宜两个数量级
    （每步只算 N×512），也不需要把 CLIP 文本塔留在计算图里。

    注意：`TEXT_PRIOR_RANDOM=True` 时 E 本身是随机向量，适配器仍然可训练，
    这会让随机文本消融"更有机会"拟合到有用方向——做消融时要意识到这一点。

    Args:
        dim: 文本嵌入维度（CLIP ViT-B/32 = 512）。
        hidden: MLP 隐层维度。
        gate_init: 残差门控初值（0 = 恒等）。
    """

    def __init__(self, dim=512, hidden=512, gate_init=0.0):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, text_embedding):
        # 文本嵌入统一用 float32 计算（AMP 下也保持稳定）
        x = text_embedding.to(torch.float32)
        return x + self.gate.to(x.dtype) * self.mlp(x)


# =============================================================================
# 2. ACM 主模块
# =============================================================================
class AdaptiveCrossModalMask(nn.Module):
    """用视觉 bottleneck 特征重建被掩码的文本语义（见文件头）。

    Args:
        in_dim: 融合前 bottleneck 的通道数（MSDA = 128）。
        text_dim: 文本嵌入维度（512）。
        mask_ratio: 文本元素被清零的比例。
        grad_to_backbone: True ⇒ 梯度回传到 MSDA 主干（默认，目标是"让主干学会
            在特征里读出文本语义"）；False ⇒ detach 输入，只训投影层与适配器。
    """

    def __init__(self, in_dim=128, text_dim=512, mask_ratio=0.15,
                 grad_to_backbone=True):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, text_dim, kernel_size=1, bias=False)
        self.mask_ratio = mask_ratio
        self.grad_to_backbone = grad_to_backbone

        self.register_buffer('text_embedding', torch.zeros(0, text_dim),
                             persistent=False)
        # 非注册引用：适配器由 model.add_module('text_adapter', ...) 注册一次，
        # 这里只持有引用，避免 state_dict 出现重名副本。
        object.__setattr__(self, '_adapter', None)

        self._feat = None           # forward_pre_hook 捕获的"融合前"特征
        self._enabled = False       # 只有训练且 ACM 生效时才捕获，省显存
        self.last_stats = {}

    # ------------------------------------------------------------------ #
    def set_text_embedding(self, text_embedding):
        self.text_embedding = text_embedding.detach().to(torch.float32)

    def hold_adapter(self, adapter):
        object.__setattr__(self, '_adapter', adapter)

    def set_enabled(self, flag):
        """由训练循环控制：λ=0 / 验证阶段关闭捕获。"""
        self._enabled = bool(flag)
        if not flag:
            self._feat = None

    # ------------------------------------------------------------------ #
    def _pre_hook(self, module, inputs):
        """挂在 `stage.layer5_4` 的 forward pre-hook：捕获其**输入**（128ch,16×16）。"""
        if not self._enabled or not self.training or not torch.is_grad_enabled():
            return
        feat = inputs[0]
        self._feat = feat if self.grad_to_backbone else feat.detach()

    # ------------------------------------------------------------------ #
    def forward_loss(self):
        """计算并返回 ACM 损失（float32 标量）；无捕获特征时返回 None。"""
        feat = self._feat
        self._feat = None                      # 立刻释放，避免跨 step 持有计算图
        if feat is None:
            return None

        B = feat.shape[0]
        # K、V 共用同一投影（与设计一致：K,V = 1×1conv(prefusion)）
        # 注意：feat 是在 autocast 下捕获的（可能是 fp16），这里统一转成投影层的
        # dtype 再算，否则 fp32 卷积权重 × fp16 输入会直接报 dtype 不匹配。
        feat = feat.to(self.proj.weight.dtype)
        V = self.proj(feat).flatten(2).transpose(1, 2).to(torch.float32)   # [B,256,512]
        V = F.normalize(V, dim=-1, eps=1e-6)

        E = self.text_embedding.to(feat.device, dtype=torch.float32)
        adapter = getattr(self, '_adapter', None)
        if adapter is not None:
            E = adapter(E)                                               # [N,512]
        En = F.normalize(E, dim=-1, eps=1e-6)

        # 元素级掩码：keep=1 保留
        keep = (torch.rand_like(En) > self.mask_ratio)
        Q = F.normalize(En * keep, dim=-1, eps=1e-6)                      # [N,512]

        d = V.shape[-1]
        attn = torch.softmax(torch.matmul(Q, V.transpose(1, 2)) / math.sqrt(d), dim=-1)
        out = torch.matmul(attn, V)                                       # [B,N,512]

        tgt = En.unsqueeze(0).expand(B, -1, -1)                           # [B,N,512]
        diff = (out - tgt) ** 2
        masked = (~keep).unsqueeze(0).expand(B, -1, -1)

        n_masked = masked.sum().clamp_min(1)
        loss_masked = diff[masked].sum() / n_masked
        with torch.no_grad():
            n_kept = (~masked).sum().clamp_min(1)
            loss_kept = diff[~masked].sum() / n_kept
        self.last_stats = dict(masked=float(loss_masked.detach()),
                               unmasked=float(loss_kept.detach()))
        return loss_masked


# =============================================================================
# 3. 零侵入接入
# =============================================================================
MODEL_ACM_CFG = {
    # 融合前 = stage.layer5_4 的输入：[64(主) + 16*4(侧路池化)] = 128 通道，16×16（256 输入）
    'MSDA': dict(layer_name='stage.layer5_4', in_dim=128),
}


def attach_cross_modal(model, text_embedding, adapter=None, choose_model='MSDA',
                       mask_ratio=0.15, grad_to_backbone=True, verbose=True):
    """把 ACM 挂到模型上：注册子模块 + 在指定层加 forward pre-hook 捕获融合前特征。

    Args:
        model: 已构建的模型（TATE 可已接入，顺序无所谓）。
        text_embedding: [N,512] 缓存的 CLIP 文本嵌入（未适配的原始 E）。
        adapter: TextAdapter 或 None；由调用方 `model.add_module` 注册一次。
        choose_model: 查 `MODEL_ACM_CFG`。
        mask_ratio: 文本元素掩码比例。
        grad_to_backbone: 是否把梯度回传到主干。
    Returns:
        (AdaptiveCrossModalMask, hook_handle)
    """
    if choose_model not in MODEL_ACM_CFG:
        raise NotImplementedError(
            f'[ACM] 模型 {choose_model} 未配置融合前特征层，请在 '
            f'components/cross_modal.py 的 MODEL_ACM_CFG 中补充。')

    cfg = MODEL_ACM_CFG[choose_model]
    acm = AdaptiveCrossModalMask(
        in_dim=cfg['in_dim'],
        text_dim=text_embedding.size(1),
        mask_ratio=mask_ratio,
        grad_to_backbone=grad_to_backbone,
    )
    acm.set_text_embedding(text_embedding)
    if adapter is not None:
        acm.hold_adapter(adapter)

    try:
        dev = next(model.parameters()).device
        acm = acm.to(dev)
    except StopIteration:
        pass

    model.add_module('cross_modal', acm)

    target = model
    for attr in cfg['layer_name'].split('.'):
        target = getattr(target, attr)
    handle = target.register_forward_pre_hook(acm._pre_hook)

    if verbose:
        n = sum(p.numel() for p in acm.parameters())
        print(f'[ACM] 已接入 {choose_model}: 捕获层输入={cfg["layer_name"]}, '
              f'in_dim={cfg["in_dim"]}→512, mask_ratio={mask_ratio}, '
              f'grad_to_backbone={grad_to_backbone}, 参数量={n} (~{n/1e3:.1f}K)')
    return acm, handle
