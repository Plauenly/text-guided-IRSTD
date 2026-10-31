# -*- coding: utf-8 -*-
"""
Gaussian-Weighted Focal Loss（替换 PAL 的 edgeSCE_loss）
=======================================================
设计目标（按需求）：
    用"点标注生成的高斯热图"作为权重，让损失把注意力集中在伪标签目标区域；
    target 仍然是**伪标签**（与 PAL 的训练语义一致）。

公式（p = sigmoid(z)，z 为 logits；g 为高斯热图 ∈[0,1]）
------------------------------------------------------------------
    target = 1（伪标签前景）:   (1 + (W-1)·g) · (1-p)^α · softplus(-z)
    target = 0（伪标签背景）:   (1-g)^γ      ·  p^α    · softplus(z)

    L = [ Σ_pos + Σ_neg ] / ( num_pos + Σ_neg (1-g)^γ )

- `log p = -softplus(-z)`、`log(1-p) = -softplus(z)`：与 CornerNet 的
  `-log(p)` 数学等价，但**全非负、不会 log(0)**，可在 AMP/fp16 下安全使用。
  （CornerNet 原版直接 log(概率)，p→0 时得 -inf；mmdet 版用 eps=1e-12 兜底，
    而 1e-12 在 fp16 下等于 0，仍会出 NaN —— 这是不复用它们的直接原因。）
- W = `pos_weight`（默认 4.0，与 PAL 原来的 edge_weight 对齐）：点处权重 4，
  越往核外越接近 1。这就是"重点关注目标区域"，且不像 edge 权重那样压在轮廓上。
- γ = `gamma`（默认 4.0）：伪标签背景在目标晕圈内被 (1-g)^γ 压小
  （CornerNet 的处理：晕圈处不该被当成硬负样本）。
- α = `alpha`（默认 2.0）：focal 的调制指数。
- 归一化 `by_pos_neg`（默认）：分母含 Σ(1-g)^γ ≈ 有效负像素数，因此
  **损失量级与 patch 尺寸/分辨率无关**，与 edgeSCE 同档（≈0.1~1），
  ACM 的 λ 无需重标定。`by_pos` 是 CornerNet 原式（量级大 3~4 个数量级，慎用）。
- `target='heatmap'`（备选）：改成 CornerNet 原式——正样本 = 热图峰值，负样本
  用 (1-g)^γ 加权。它**能抑制伪标签膨胀**（对非峰值的大块高置信重罚），
  但与"PAL 用伪标签训练"的语义不一致，仅作对照。

用法（train_model.py）
----------------------
    loss_fn = GaussianFocalLoss(alpha=2.0, gamma=4.0, pos_weight=4.0,
                                norm='by_pos_neg', target='pseudo')
    loss = loss_fn(logits, pseudo_mask, heatmap)     # 与 edgeSCE_loss 同签名
    loss_fn.last_stats                               # 本 batch 的统计量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFocalLoss(nn.Module):
    """见文件头。`forward(logits, target, heatmap)`，与 `edgeSCE_loss` 签名一致。

    Args:
        alpha: focal 调制指数（正负样本各自的 (1-p)^α / p^α）。
        gamma: 负样本的高斯衰减指数 (1-g)^γ。
        pos_weight: 核内正向最大权重 W（g=1 处）。
        norm: 'by_pos_neg'（默认）| 'by_area' | 'by_pos'。
        target: 'pseudo'（用伪标签当 target，默认）| 'heatmap'（CornerNet 原式）。
        pos_eps: 判"热图峰值像素"的容差（仅 target='heatmap' 用）。
        skip_empty: 无正样本的样本不计入损失（随机裁剪可能把目标裁掉）。
        stats: 是否记录 last_stats。
    """

    def __init__(self, alpha=2.0, gamma=4.0, pos_weight=4.0,
                 norm='by_pos_neg', target='pseudo', pos_eps=1e-4,
                 skip_empty=True, stats=True):
        super().__init__()
        assert norm in ('by_pos_neg', 'by_area', 'by_pos')
        assert target in ('pseudo', 'heatmap')
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.pos_weight = float(pos_weight)
        self.norm = norm
        self.target = target
        self.pos_eps = float(pos_eps)
        self.skip_empty = bool(skip_empty)
        self.collect_stats = bool(stats)
        self.last_stats = {}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_4d(t):
        """统一成 [B,1,H,W]。"""
        if t.dim() == 3:
            t = t.unsqueeze(1)
        return t

    def forward(self, logits, target, heatmap):
        # 兼容深监督模型（DNA/UIU 返回 list/tuple）：逐输出算损失后取均值。
        # 当前 MSDA 返回单个张量，走不到这个分支。
        if isinstance(logits, (list, tuple)):
            return sum(self.forward(l, target, heatmap) for l in logits) / len(logits)

        z = self._to_4d(logits).float()
        tgt = self._to_4d(target.float())
        g = self._to_4d(heatmap.float()).clamp(0.0, 1.0)

        if z.shape[-2:] != tgt.shape[-2:]:
            # 防御：分辨率不一致时把目标/热图对齐到预测（最近邻，保持二值）
            tgt = F.interpolate(tgt, size=z.shape[-2:], mode='nearest')
            g = F.interpolate(g, size=z.shape[-2:], mode='bilinear', align_corners=False)

        p = torch.sigmoid(z)
        B = z.shape[0]

        if self.target == 'pseudo':
            pos = (tgt > 0.5).float()
            w_pos = 1.0 + (self.pos_weight - 1.0) * g
        else:
            # CornerNet 原式：正样本是热图峰值，峰值为 1 的像素才计入正样本
            pos = (g > 1.0 - self.pos_eps).float()
            w_pos = torch.ones_like(g)
        neg = 1.0 - pos
        w_neg = (1.0 - g).pow(self.gamma)

        # log p = -softplus(-z), log(1-p) = -softplus(z)：等价的稳定写法
        pos_term = w_pos * (1.0 - p).pow(self.alpha) * F.softplus(-z)
        neg_term = w_neg * p.pow(self.alpha) * F.softplus(z)

        pos_sum = (pos_term * pos).flatten(1).sum(1)          # [B]
        neg_sum = (neg_term * neg).flatten(1).sum(1)          # [B]
        num_pos = pos.flatten(1).sum(1)                       # [B]
        neg_w_sum = (w_neg * neg).flatten(1).sum(1)           # [B]

        if self.norm == 'by_pos':
            denom = num_pos.clamp_min(1.0)
        elif self.norm == 'by_area':
            denom = torch.full_like(num_pos, float(z.shape[-2] * z.shape[-1]))
        else:                                                  # by_pos_neg
            denom = (num_pos + neg_w_sum).clamp_min(1.0)

        loss_per = (pos_sum + neg_sum) / denom                 # [B]

        valid = num_pos > 0 if self.skip_empty else torch.ones_like(num_pos, dtype=torch.bool)
        if valid.any():
            loss = loss_per[valid].mean()
        else:
            # 整个 batch 都没有目标：给一个需要梯度的 0，保持计算图闭合
            loss = (pos_sum + neg_sum).sum() * 0.0

        if self.collect_stats:
            with torch.no_grad():
                fg = pos > 0.5
                n_fg = fg.flatten(1).sum(1).clamp_min(1.0)
                mean_p_pos = ((p * fg).flatten(1).sum(1) / n_fg)[valid].mean() if valid.any() else p.new_tensor(0.)
                mean_p_neg = (p * neg).flatten(1).sum(1) / neg.flatten(1).sum(1).clamp_min(1.0)
                mean_p_neg = mean_p_neg[valid].mean() if valid.any() else p.new_tensor(0.)
                self.last_stats = dict(
                    task_loss=float(loss.detach()),
                    mean_p_pos=float(mean_p_pos),
                    mean_p_neg=float(mean_p_neg),
                    pos_px=float(num_pos[valid].mean()) if valid.any() else 0.0,
                    # 预测前景面积（p>0.5 的像素数），用来监控伪标签/预测是否膨胀
                    pred_fg_px=float((p > 0.5).float().flatten(1).sum(1).float().median()),
                    empty_ratio=float((~valid).float().mean()),
                )
        return loss
