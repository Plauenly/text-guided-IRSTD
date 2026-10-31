# -*- coding: utf-8 -*-
"""
高斯热图生成器自检（只依赖 numpy/cv2，可在无 torch 环境直接跑）

    python tools/test_heatmap.py

覆盖：
    1 单点：峰值恰为 1、形状对称、3σ 外≈0、取值∈[0,1]
    2 多点：按 max 合并（不会 >1），每个点都有峰
    3 空标注：全 0
    4 边界点：不越界、峰值=1
    5 from_label 模式：σ 随连通域面积变化并被 clamp
    6 softplus 恒等式（loss 用到的稳定写法）数值验证
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.heatmap import gaussian_heatmap, point_centers  # noqa: E402


def _peaks(mask):
    return point_centers(mask)


def main():
    H = W = 64

    # ---------- 1. 单点 ----------
    pm = np.zeros((H, W), np.uint8)
    pm[30, 32] = 255
    g = gaussian_heatmap(pm, sigma=1.5)
    assert g.dtype == np.float32 and g.shape == (H, W)
    assert g[30, 32] == 1.0, f'峰值必须严格为 1，实际 {g[30, 32]}'
    assert g.min() >= 0.0 and g.max() <= 1.0
    assert np.allclose(g[30, 32 - 1], g[30, 32 + 1]) and np.allclose(g[29, 32], g[31, 32])
    # 3σ(=4.5→半径5) 之外应几乎为 0
    assert g[30, 32 + 8] < 1e-3, g[30, 32 + 8]
    # 半高宽：1 像素之外应约 exp(-1/(2*1.5^2)) ≈ 0.80
    assert abs(g[30, 33] - np.exp(-1 / (2 * 1.5 ** 2))) < 1e-5
    print(f'[1] 单点 OK: 峰值={g[30,32]:.3f} 相邻={g[30,33]:.4f} 8px外={g[30,40]:.2e}')

    # ---------- 2. 多点 max 合并 ----------
    pm2 = np.zeros((H, W), np.uint8)
    pm2[30, 32] = 255
    pm2[30, 34] = 255          # 间距 2 px：中点 (30,33) 同时落在两个核内
    g2 = gaussian_heatmap(pm2, sigma=1.5)
    assert len(_peaks(pm2)) == 2
    assert g2.max() <= 1.0 + 1e-6, g2.max()
    assert g2[30, 32] == 1.0 and g2[30, 34] == 1.0
    # 中点：每个高斯贡献 exp(-1/4.5)≈0.801；相加会是 1.6，max 合并必须是 0.801
    expect = np.exp(-1.0 / (2 * 1.5 ** 2))
    mid = g2[30, 33]
    assert abs(mid - expect) < 1e-6, (mid, expect)
    assert mid < 1.0, '中点等于 1 说明两个高斯被相加了'
    print(f'[2] 多点 max 合并 OK: 峰1={g2[30,32]:.3f} 峰2={g2[30,34]:.3f} 中点={mid:.4f}(相加会=%.3f)' % (2*expect))

    # ---------- 3. 空标注 ----------
    g3 = gaussian_heatmap(np.zeros((H, W), np.uint8), sigma=1.5)
    assert g3.sum() == 0.0
    print('[3] 空标注 OK: 全 0')

    # ---------- 4. 边界点 ----------
    for (y, x) in [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1), (0, 32)]:
        pm4 = np.zeros((H, W), np.uint8)
        pm4[y, x] = 255
        g4 = gaussian_heatmap(pm4, sigma=1.5)
        assert g4[y, x] == 1.0, (y, x, g4[y, x])
    print('[4] 边界点 OK: 裁剪不越界且峰值=1')

    # ---------- 5. from_label 模式 ----------
    lab = np.zeros((H, W), np.float32)
    lab[28:35, 30:37] = 1.0                 # 7×7 = 49 px
    s_ref = 0.5 * np.sqrt(49 / np.pi)       # ≈1.97
    pm5 = np.zeros((H, W), np.uint8)
    pm5[31, 33] = 255
    g5 = gaussian_heatmap(pm5, sigma_mode='from_label', label_mask=lab,
                          sigma_min=1.0, sigma_max=4.0)
    assert g5[31, 33] == 1.0
    # 与固定 σ=1.97 的结果应一致
    g5b = gaussian_heatmap(pm5, sigma=s_ref)
    assert np.abs(g5 - g5b).max() < 1e-6, np.abs(g5 - g5b).max()
    # clamp 检查：大面积标签 → σ 被截到 max
    lab_big = np.zeros((H, W), np.float32)
    lab_big[5:60, 5:60] = 1.0
    pm5[31, 33] = 255
    g5c = gaussian_heatmap(pm5, sigma_mode='from_label', label_mask=lab_big,
                           sigma_min=1.0, sigma_max=4.0)
    g5d = gaussian_heatmap(pm5, sigma=4.0)
    assert np.abs(g5c - g5d).max() < 1e-6
    print(f'[5] from_label OK: 49px→σ={s_ref:.2f}；大标签被截到 σ_max=4.0')

    # ---------- 6. softplus 恒等式 ----------
    # loss 里用 log p = -softplus(-z)、log(1-p) = -softplus(z) 取代直接取 log：
    # 数值上等价，但 z 很大时不会 log(0)。
    z = np.array([-10, -5, -1, 0, 1, 5, 10], np.float64)
    p = 1.0 / (1.0 + np.exp(-z))
    softplus = lambda v: np.logaddexp(0.0, v)      # = log(1+e^v)，数值稳定
    assert np.allclose(-softplus(-z), np.log(p), atol=1e-9)
    assert np.allclose(-softplus(z), np.log1p(-p), atol=1e-9)
    print('[6] softplus 恒等式 OK（|z|<=10）')

    # z=±40：朴素 log(p) 会因 sigmoid 饱和成 -inf / 1.0 而失败，稳定写法仍精确
    zz = np.array([-40.0, 40.0])
    stable = -softplus(zz)
    assert np.all(np.isfinite(stable)) and abs(stable[1] + 40.0) < 1e-9   # log(1-p)≈-40
    naive = np.log(1.0 / (1.0 + np.exp(-zz)))
    with np.errstate(divide='ignore'):          # 故意演示朴素写法会出 -inf
        naive2 = np.log1p(-1.0 / (1.0 + np.exp(-zz)))
    print(f'[7] 饱和区 OK: z=-40/40 时稳定写法={stable}; 朴素 log1p(1-p)={naive2}'
          f'（后者在 z=-40 时为 -inf）')

    print('\n全部自检通过。')


if __name__ == '__main__':
    main()
