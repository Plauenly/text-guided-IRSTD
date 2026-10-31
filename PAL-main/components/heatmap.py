# -*- coding: utf-8 -*-
"""
由点标注生成高斯热图（Gaussian heatmap）
=======================================
用途：给 Gaussian-Focal-Loss 提供"目标区域权重图" g ∈ [0,1]。

- 每个标注点画一个各向同性高斯，**多目标取 max**（不是相加，否则相邻目标会让 g>1，
  把正/负样本的划分与 (1-g)^4 权重都搞乱）。
- 峰值强制为 **1.0**（exp(0)=1，但浮点/插值可能给出 0.99999994；CornerNet 用严格
  `gt==1` 判正样本，差一点就会让 num_pos=0 → 损失走到无归一化分支，梯度爆炸）。
- 核半径取 `3σ`（3σ 外 g<0.011，可忽略），按图像边界裁剪。

σ 的取值（点标注没有尺寸信息，只能按经验定）
    fixed      : σ = SIGMA_PX（默认 1.5，对应 3~9 px 的红外小目标核心）
    from_label : σ = clip(0.5·√(A/π), MIN, MAX)，A = 该点所在伪标签连通域面积
                 （仅当伪标签可信时使用；MAX 用于防止 33×33 方块把 σ 撑大）

本文件不依赖 torch，可直接在无 GPU 环境做数值单测（见 tools/test_heatmap.py）。
"""

import numpy as np
import cv2


def point_centers(point_mask, thr=127):
    """取出点标注的中心坐标（对 1 px 点和稍粗的点都稳）。

    Args:
        point_mask: [H,W] 点标注图，非零/大于 thr 的像素视为点。
    Returns:
        list[(x, y)]，按行列扫描顺序。
    """
    p = (point_mask > thr).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(p, connectivity=8)
    centers = []
    for i in range(1, n):                      # 0 是背景
        cx, cy = cents[i]
        centers.append((int(round(cx)), int(round(cy))))
    return centers


def _draw_gaussian(heatmap, cx, cy, sigma, k=3.0, min_val=0.01):
    """在 heatmap 上以 (cx,cy) 为中心画一个高斯，取 max 合并，峰值置 1。"""
    H, W = heatmap.shape
    if sigma <= 0:
        return
    radius = max(1, int(np.ceil(k * sigma)))
    x0, x1 = max(0, cx - radius), min(W, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(H, cy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys = np.arange(y0, y1, dtype=np.float32) - cy
    xs = np.arange(x0, x1, dtype=np.float32) - cx
    gy = np.exp(-(ys ** 2) / (2.0 * sigma ** 2))
    gx = np.exp(-(xs ** 2) / (2.0 * sigma ** 2))
    g = np.outer(gy, gx)
    g[g < min_val] = 0.0
    np.maximum(heatmap[y0:y1, x0:x1], g, out=heatmap[y0:y1, x0:x1])
    heatmap[cy, cx] = 1.0                      # 峰值强制为 1（见文件头说明）


def _sigma_from_label(label_mask, cx, cy, sigma_min, sigma_max):
    """按"该点所在连通域的面积"折算 σ：σ = clip(0.5·√(A/π), min, max)。"""
    n, labels = cv2.connectedComponents((label_mask > 0.5).astype(np.uint8), connectivity=8)
    lbl = labels[cy, cx]
    if lbl == 0:                               # 点不在任何前景连通域里
        return sigma_min
    area = float((labels == lbl).sum())
    return float(np.clip(0.5 * np.sqrt(area / np.pi), sigma_min, sigma_max))


def gaussian_heatmap(point_mask, sigma=1.5, sigma_mode='fixed',
                     label_mask=None, sigma_min=1.0, sigma_max=4.0, k=3.0):
    """由点标注生成高斯热图。

    Args:
        point_mask: [H,W] 点标注图（>127 视为点）。
        sigma: fixed 模式下的 σ（像素，位于"最终送入网络的分辨率"，即 patch 空间）。
        sigma_mode: 'fixed' 或 'from_label'。
        label_mask: [H,W] 伪标签（0/1 或 0/255），from_label 模式必需。
        sigma_min / sigma_max: from_label 模式的裁剪范围。
        k: 核半径 = k·σ。
    Returns:
        [H,W] float32，取值 [0,1]，峰值 1。无点时返回全 0。
    """
    H, W = point_mask.shape
    heat = np.zeros((H, W), dtype=np.float32)
    centers = point_centers(point_mask)
    if len(centers) == 0:
        return heat

    for (cx, cy) in centers:
        if sigma_mode == 'from_label' and label_mask is not None:
            s = _sigma_from_label(label_mask, cx, cy, sigma_min, sigma_max)
        else:
            s = float(sigma)
        _draw_gaussian(heat, cx, cy, s, k=k)

    np.clip(heat, 0.0, 1.0, out=heat)
    return heat
