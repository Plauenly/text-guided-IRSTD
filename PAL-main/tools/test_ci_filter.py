# -*- coding: utf-8 -*-
"""
CI 门控自检脚本（components/credibility.py）

不依赖数据集、不依赖 CLIP，覆盖：
    1. 连通域提取 / 过点匹配 / 虚警比
    2. 置信度 S 与 CI 公式
    3. GMM 动态阈值（SeViL gmm_policy 的 policy='high'）
    4. 图级均值 CI 的"漏点惩罚"

加 --with_clip 再额外验证 CLIP 图像塔 + 局部窗裁剪（需要一个可用权重）：
    python tools/test_ci_filter.py --with_clip --clip_path /path/ViT-B-32.pt
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.credibility import (  # noqa: E402
    component_confidence, credibility_index, extract_components,
    spurious_ratio, gmm_threshold, fallback_threshold, sample_mean_ci,
    admit_ratio,
)


def make_case(H=64, W=64, comps=((20, 20, 24, 24), (44, 40, 48, 44)),
              points=((21, 21), (45, 41)), probs=(0.9, 0.6)):
    """构造一张合成图：comps 为 (x0,y0,x1,y1) 的矩形连通域，points 为 (x,y)。"""
    prob = np.zeros((H, W), dtype=np.float32)
    for (x0, y0, x1, y1), p in zip(comps, probs):
        prob[y0:y1, x0:x1] = p
    pm = np.zeros((H, W), dtype=np.uint8)
    for x, y in points:
        pm[y, x] = 255
    return prob, pm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_clip', action='store_true')
    parser.add_argument('--clip_path', default='ViT-B/32')
    args = parser.parse_args()

    # ---------- 1. 组件提取与过点匹配 ----------
    prob, points = make_case()
    info = extract_components(prob, points)
    assert info['n_points'] == 2, info['n_points']
    assert info['n_hit'] == 2, info['n_hit']
    assert info['n_spurious'] == 0, info['n_spurious']
    assert abs(spurious_ratio(0, 2)) < 1e-9
    print(f'[1] 组件提取 OK: hit={info["n_hit"]} spurious={info["n_spurious"]} '
          f'points={info["n_points"]} point_comp={info["point_comp"]}')

    # 多加一个不过点的连通域 -> 变成虚警，且虚警比 = 1/2
    prob2, points2 = make_case(comps=((20, 20, 24, 24), (44, 40, 48, 44), (4, 50, 8, 54)),
                              points=((21, 21), (45, 41)), probs=(0.9, 0.6, 0.95))
    info2 = extract_components(prob2, points2)
    assert info2['n_hit'] == 2 and info2['n_spurious'] == 1, info2
    assert abs(spurious_ratio(1, 2) - 0.5) < 1e-9
    print(f'[2] 虚警比 OK: hit=2 spurious=1 ratio={spurious_ratio(1, 2):.3f}')

    # ---------- 2. S 与 CI ----------
    S = component_confidence(prob, info['comps'][1]['mask'])
    assert abs(S - 0.9) < 1e-5, S
    assert abs(credibility_index(0.9, 0.9) - 0.9) < 1e-5
    assert abs(credibility_index(0.9, 0.2) - 2 * 0.9 * 0.2 / 1.1) < 1e-6
    print(f'[3] S/CI OK: S={S:.4f} CI(0.9,0.2)={credibility_index(0.9, 0.2):.4f}')

    # ---------- 3. 漏点惩罚：一个点漏掉时 mean_ci 减半 ----------
    points_half = np.zeros_like(points)
    points_half[21, 21] = 255          # 只留一个点，另一个点没有组件覆盖
    info3 = extract_components(prob, points_half)
    # 用一个假的 scorer 返回固定 K，避免依赖 CLIP
    class _FakeScorer:
        def score_components(self, gray, bboxes):
            return np.full(len(bboxes), 0.5, dtype=np.float32)

    gray = (prob * 255).astype(np.uint8)
    ci = sample_mean_ci(info3, gray, prob, _FakeScorer())
    # 单点且命中 -> mean_ci = 该点 CI
    assert abs(ci['mean_ci'] - credibility_index(0.9, 0.5)) < 1e-6, ci
    # 两点但只命中一点 -> mean_ci 被未命中点（CI=0）拉低一半
    ci_full = sample_mean_ci(info, gray, prob, _FakeScorer())
    expect = (credibility_index(0.9, 0.5) + credibility_index(0.6, 0.5)) / 2
    assert abs(ci_full['mean_ci'] - expect) < 1e-6, (ci_full, expect)
    print(f'[4] 漏点惩罚 OK: 单点={ci["mean_ci"]:.4f} 两点={ci_full["mean_ci"]:.4f}')

    # ---------- 4. GMM 阈值 ----------
    # 均匀双峰：低簇 0.1~0.2，高簇 0.6~0.7 -> theta 应落在高簇内（约高簇中心）
    vals = list(np.linspace(0.10, 0.20, 60)) + list(np.linspace(0.60, 0.70, 60))
    theta = gmm_threshold(vals)
    assert theta is not None and 0.60 <= theta <= 0.70, theta
    print(f'[5] GMM 阈值 OK: theta={theta:.4f}（低簇 max=0.20, 高簇 0.60~0.70）')

    # 真实双峰的放行率应在 ~1/4 量级（这是"节流"的关键）：
    # 若误用 SeViL 的 precisions_init=1.0，分量会重叠、阈值退回分布中心，放行 ~50%
    rng = np.random.default_rng(0)
    vals2 = np.r_[rng.normal(0.15, 0.03, 300), rng.normal(0.55, 0.05, 300)].clip(0, 1)
    theta2 = gmm_threshold(vals2)
    admit = float(np.mean(vals2 >= theta2))
    assert 0.15 <= admit <= 0.35, f'放行率 {admit:.2f} 不在 15%~35%，GMM 可能退化了'
    print(f'[6] 双峰放行率 OK: theta={theta2:.4f} 放行={admit:.0%}')

    # 样本不足 -> None -> 回退中位数
    assert gmm_threshold([0.3, 0.4, 0.5]) is None
    assert abs(fallback_threshold([0.1, 0.2, 0.3, 0.4], 'median') - 0.25) < 1e-9
    # 退化输入不应崩（全相等 / 含零堆积）
    assert gmm_threshold([0.5] * 20) is not None
    assert gmm_threshold([0.0] * 200 + [0.5] * 200) is not None
    print('[7] 少量样本回退 + 退化输入保护 OK')

    # ---------- 5. 数量闸的比例插值 ----------
    r_start = admit_ratio(int(400 * 0.2) + 1, 400, 0.05, 0.2, 0.8)
    r_mid = admit_ratio(200, 400, 0.05, 0.2, 0.8)
    r_end = admit_ratio(320, 400, 0.05, 0.2, 0.8)
    assert 0.05 < r_start < r_mid < r_end, (r_start, r_mid, r_end)
    assert abs(r_end - 1.0) < 1e-6, r_end          # enhancement 末尾 = 不限制
    assert abs(admit_ratio(0, 400, 0.05, 0.2, 0.8) - 0.05) < 1e-9  # 早于阶段也不越界
    print(f'[8] 数量闸比例 OK: start={r_start:.3f} mid={r_mid:.3f} end={r_end:.3f}')

    # ---------- 6. 可选：CLIP 图像塔 ----------
    if args.with_clip:
        import torch
        from components.text_prompts import SINGLE_FRAME_IR_PROMPTS
        from components.text_prior import ClipTextEmbedder
        from components.credibility import CIPatchScorer

        emb = ClipTextEmbedder(SINGLE_FRAME_IR_PROMPTS, args.clip_path,
                               device='cpu', keep_image_tower=True, verbose=False)
        scorer = CIPatchScorer(emb.get_image_tower(), emb.get(), 'cpu')
        K = scorer.score_components(gray, [info['comps'][l]['bbox'] for l in sorted(info['comps'])])
        assert K.shape == (2,), K.shape
        assert np.all(np.isfinite(K))
        print(f'[9] CLIP K OK: {K}')

    print('\n全部自检通过。')


if __name__ == '__main__':
    main()
