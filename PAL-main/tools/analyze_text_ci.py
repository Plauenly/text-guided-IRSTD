# -*- coding: utf-8 -*-
"""
Step-0 离线体检：文本 prompt / K 的可分性评估（不需要训练，不需要 GPU 也能跑）

目的：在改 prompt、加适配器/ACM 之前，先回答两个问题——
  Q1 新 prompt 是否把 K 的"相对波动"显著拉大？（K 全距只有 0.207~0.266 时，阈值没意义）
  Q2 K 是否与"目标可见性"相关？（用 GT 计算的弱标签，不需要人工标注）

用法：
    python tools/analyze_text_ci.py --n 300
    python tools/analyze_text_ci.py --n 300 --prompts_file /path/my_prompts.py --var MY_PROMPTS
    # 同时对比两套 prompt：
    python tools/analyze_text_ci.py --n 300 --prompts_file /path/my_prompts.py --var MY_PROMPTS --compare

输出：各统计量 + 每个 crop 的明细 CSV（便于你自己看）。
"""

import argparse
import importlib.util
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.credibility import CIPatchScorer          # noqa: E402


def load_prompt_module(path, var):
    spec = importlib.util.spec_from_file_location('custom_prompts', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(getattr(mod, var))


def gt_quality(gray, gt_mask, point):
    """不依赖人工标注的"目标可见性"弱标签（越大约明显）。"""
    y, x = point
    g = gray.astype(np.float32)
    bg = g[~gt_mask]
    if len(bg) < 10 or gt_mask.sum() < 1:
        return np.nan, np.nan
    contrast = (g[gt_mask].mean() - np.median(bg)) / (bg.std() + 1e-6)
    # 点处 3x3 峰值 vs 局部环背景
    R = 18
    y0, y1 = max(0, y - R), min(g.shape[0], y + R + 1)
    x0, x1 = max(0, x - R), min(g.shape[1], x + R + 1)
    big = g[y0:y1, x0:x1]
    yy = np.arange(y0, y1)[:, None]
    xx = np.arange(x0, x1)[None, :]
    ring = big[~((np.abs(yy - y) <= 2) & (np.abs(xx - x) <= 2))]
    snr = (g[max(0, y - 1):y + 2, max(0, x - 1):x + 2].max() - np.median(ring)) / (ring.std() + 1e-6)
    return float(contrast), float(snr)


def eval_prompts(prompts, scorer, samples, gray_cache, tag, csv_rows):
    kt, kb, deltas = [], [], []
    for s in samples:
        g = gray_cache[s['name']]
        boxes_t = [tuple(s['box'])]
        # 随机背景窗（与目标窗同尺寸、不与 GT 重叠）
        h, w = g.shape
        side = max(s['box'][2] - s['box'][0], s['box'][3] - s['box'][1])
        side = max(2, min(side, min(h, w) - 1))
        box_b = s['box']
        for _ in range(50):
            cx = int(np.random.randint(0, max(1, w - side)))
            cy = int(np.random.randint(0, max(1, h - side)))
            cand = (cx, cy, min(w, cx + side), min(h, cy + side))
            bx0, by0, bx1, by1 = cand
            if not s['gt'][by0:by1, bx0:bx1].any():
                box_b = cand
                break
        k_t = scorer.score_components(g, boxes_t)[0]
        k_b = scorer.score_components(g, [box_b])[0]
        kt.append(float(k_t)); kb.append(float(k_b)); deltas.append(float(k_t - k_b))
    kt = np.array(kt); kb = np.array(kb); dl = np.array(deltas)
    print(f'\n--- prompt 集 [{tag}] ---')
    print(f'  K_target : mean={kt.mean():.4f} std={kt.std():.4f}  '
          f'CV={100*kt.std()/kt.mean():.1f}%  全距={kt.max()-kt.min():.4f} '
          f'({kt.max()/max(1e-9,kt.min()):.2f}x)')
    print(f'  K_bg     : mean={kb.mean():.4f} std={kb.std():.4f}  CV={100*kb.std()/kb.mean():.1f}%')
    print(f'  Δ=K_t-K_b: mean={dl.mean():.4f} std={dl.std():.4f}  '
          f'CV={100*dl.std()/abs(dl.mean()):.1f}%  Δ>0 占比={100*np.mean(dl>0):.0f}%')
    for i, s in enumerate(samples):
        csv_rows.append([tag, s['name'], round(kt[i], 6), round(kb[i], 6), round(dl[i], 6),
                         round(s['contrast'], 4), round(s['snr'], 4)])
    return kt, kb, dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='dataset/SIRST3/origin')
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--clip_path', default='ViT-B/32')
    ap.add_argument('--prompts_file', default=None, help='自定义 prompt 的 .py 文件')
    ap.add_argument('--var', default='MY_PROMPTS', help='prompt 列表的变量名')
    ap.add_argument('--compare', action='store_true', help='同时评估内置 prompt，做对比')
    ap.add_argument('--out', default='work_dirs/text_ci_report.csv')
    args = ap.parse_args()

    import csv
    from components.text_prior import ClipTextEmbedder

    img_dir = os.path.join(args.dataset, 'img')
    gt_dir = os.path.join(args.dataset, 'mask')
    pt_dir = os.path.join(args.dataset, 'masks_coarse')

    names = sorted(os.listdir(img_dir))
    rng = np.random.default_rng(0)
    names = list(rng.choice(names, size=min(args.n, len(names)), replace=False))

    samples, gray_cache = [], {}
    for n in names:
        g = cv2.imread(os.path.join(img_dir, n), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(os.path.join(gt_dir, n), cv2.IMREAD_GRAYSCALE)
        pt = cv2.imread(os.path.join(pt_dir, n), cv2.IMREAD_GRAYSCALE)
        if g is None or gt is None or pt is None:
            continue
        ys, xs = np.where(pt > 127)
        if len(ys) == 0:
            continue
        y, x = int(ys[0]), int(xs[0])
        R = max(8, int(round(0.75 * 16)))
        box = (max(0, x - R), max(0, y - R), min(g.shape[1], x + R), min(g.shape[0], y + R))
        ctr, snr = gt_quality(g, gt > 127, (y, x))
        gray_cache[n] = g
        samples.append(dict(name=n, box=box, gt=gt > 127, contrast=ctr, snr=snr, point=(y, x)))
    print(f'样本数={len(samples)}')

    rows = []
    prompt_sets = []
    if args.prompts_file:
        prompt_sets.append((args.var, load_prompt_module(args.prompts_file, args.var)))
    if args.compare or not args.prompts_file:
        from components.text_prompts import SINGLE_FRAME_IR_PROMPTS
        prompt_sets.append(('builtin', list(SINGLE_FRAME_IR_PROMPTS)))

    results = {}
    for tag, prompts in prompt_sets:
        emb = ClipTextEmbedder(prompts, args.clip_path, device='cpu',
                               keep_image_tower=True, verbose=False)
        # prompt 集的秩（重建任务是否有区分度）
        E = emb.get().numpy()
        Ec = E - E.mean(0, keepdims=True)
        sv = np.linalg.svd(Ec, compute_uv=False)
        print(f'[秩] {tag}: 第一奇异值能量占比={sv[0]**2/(sv**2).sum():.3f} '
              f'（>0.95 ⇒ 10 条近乎同义，ACM 重建会退化）')
        scorer = CIPatchScorer(emb.get_image_tower(), emb.get(), 'cpu')
        results[tag] = eval_prompts(prompts, scorer, samples, gray_cache, tag, rows)
        del scorer, emb

    # 两套 prompt 的 K 排序一致性 + 与可见性弱标签的相关性
    if len(results) == 2:
        from scipy.stats import spearmanr
        keys = list(results.keys())
        t1, t2 = keys[0], keys[1]
        kt1, kt2 = results[t1][0], results[t2][0]
        print(f'\n[对比] Spearman(K({t1}), K({t2})) = {spearmanr(kt1, kt2).statistic:.3f}')
    ctr = np.array([s['contrast'] for s in samples])
    snr = np.array([s['snr'] for s in samples])
    from scipy.stats import spearmanr
    for tag, (kt, kb, dl) in results.items():
        ok = np.isfinite(ctr) & np.isfinite(snr)
        print(f'[相关性] {tag}: Spearman(K_t, GT对比度)={spearmanr(kt[ok], ctr[ok]).statistic:.3f}  '
              f'Spearman(K_t, 局部SNR)={spearmanr(kt[ok], snr[ok]).statistic:.3f}  '
              f'Spearman(Δ, GT对比度)={spearmanr(dl[ok], ctr[ok]).statistic:.3f}')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['prompt_set', 'name', 'K_target', 'K_bg', 'delta', 'gt_contrast', 'local_snr'])
        w.writerows(rows)
    print(f'\n明细已写：{args.out}')


if __name__ == '__main__':
    main()
