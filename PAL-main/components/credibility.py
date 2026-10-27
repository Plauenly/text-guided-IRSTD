# -*- coding: utf-8 -*-
"""
CI（Credibility Index，可信度指数）门控
======================================
用 SeViL(AAAI'26, TAPF) 的"CI + GMM 动态阈值"机制，替换 PAL 的困难样筛选判据
（原 `utilts.nc_pred_mask` + `deal_pred_mask_and_true_point_in`）。

--- 为什么要换 ---------------------------------------------------------------
PAL 原判据是纯几何的：点位漏检率 <= lose_point_ratio 且 虚警比例 <=
alarm_point_ration，就整张样本进简单池。阈值人工给定、样本级一刀切，
实测每一轮放行过多 → 伪标签池突变 → 训练震荡。

SeViL 的 TAPF（`SeViL/projects/MixPL/mixpl/mixpl.py` 的
`loss_by_pseudo_instances` / `gmm_policy`）是实例级 + 自适应的：
    CI = 2*S*K / (S + K)        # S=检测置信度, K=图文余弦相似度
    对所有 CI 拟合 2 分量 GMM，取高簇密度峰值处的 CI 作为动态阈值。
本文件把该机制改造成"单帧红外 + 点监督 + 无检测头"的形式。

--- 改造点（与原版的差异，均为必要差异）--------------------------------------
1) S：红外分割网没有 bbox score，只有稠密概率图，故
       S = 0.5*max(prob[comp]) + 0.5*mean(prob[comp])
2) K：红外目标只有几个像素，直接按 bbox 裁剪再 resize 放大会退化成插值噪声，
   故改为"以连通域中心取正方形局部窗"（边长 = 1.5*bbox 边长，夹在
   [CI_CROP_MIN, CI_CROP_MAX] 内），再 resize 到 224 送 CLIP 图像塔。
3) 统计单元：SeViL 是"实例级"，PAL 需要"样本级"决定谁进简单池，故
       mean_ci(图) = mean(该图每个点对应组件的 CI)，未匹配到组件的点记 CI=0。
   因此 mean_ci 天然包含"漏点惩罚"（原 PAL 靠 lose_point_ratio 爬坡管的事），
   不需要额外的召回率条件。
4) 阈值：对"全部通过虚警预筛的困难样本"的 mean_ci 拟合 GMM(2)，取高簇
   密度峰值处的值；`gmm_policy` 的 "high" 逻辑与 SeViL 完全一致。

--- 判决规则（唯一判据：theta）---------------------------------------------
    mean_ci >= theta  ->  进简单池（随后由 `deal_gen_mask_error_aera` 只保留
                          过点连通域并与点标注融合，虚警一律丢弃）
    mean_ci <  theta  ->  留在困难池
其余细节见 PAL+TATE/CI_FILTER_CHANGELOG.md。
"""

import cv2
import warnings
import numpy as np
import torch
from sklearn.mixture import GaussianMixture

# CLIP 的图像归一化参数（与 openai clip 的 preprocess 一致）
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


# =============================================================================
# 1. 纯函数：置信度 S、可信度指数 CI
# =============================================================================
def component_confidence(prob, comp_mask):
    """连通域置信度 S（替代 SeViL 里的检测框 confidence）。

    Args:
        prob: [H,W] float32，sigmoid 之后的概率图。
        comp_mask: [H,W] bool，该连通域的像素。
    Returns:
        float，0.5*max + 0.5*mean（既奖励峰值高，也奖励内部一致/面积实）。
    """
    vals = prob[comp_mask]
    return 0.5 * float(vals.max()) + 0.5 * float(vals.mean())


def credibility_index(S, K, eps=1e-8):
    """CI = 2SK/(S+K)，与 SeViL 的调和平均完全一致。"""
    return 2.0 * S * K / (S + K + eps)


# =============================================================================
# 2. 组件提取 + 点匹配（阶段 A：纯几何，不需要 CLIP）
# =============================================================================
def extract_components(pred_prob, point_mask, prob_thr=0.5, connectivity=8):
    """把预测概率图拆成连通域，并标注哪些连通域"过点"。

    Args:
        pred_prob: [H,W] float32，原始分辨率下的概率图。
        point_mask: [H,W] uint8，点标注（255 = 点）。
        prob_thr: 二值化阈值。
    Returns:
        dict，字段：
            bin_mask    [H,W] uint8，二值预测（0/1），用于落盘与后续精炼
            n_points    int，标注点个数
            n_hit       int，至少覆盖一个点的连通域个数（"过点组件"）
            n_spurious  int，一个点都不覆盖的连通域个数（疑似虚警）
            point_comp  list[int]，每个点落在的组件 label；未落在任何组件上为 0
            comps       dict[label] -> dict(mask=bool[H,W], bbox=(x0,y0,x1,y1))
    """
    H, W = pred_prob.shape
    if point_mask.shape != pred_prob.shape:
        # 防御：点标注与预测尺寸不一致时对齐（最近邻，避免引入灰边）
        point_mask = cv2.resize(point_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    bin_mask = (pred_prob > prob_thr).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(bin_mask, connectivity=connectivity)

    # 每个点是 (y, x)，取该像素所属组件 label
    ys, xs = np.where(point_mask > 127)
    n_points = int(len(ys))
    point_comp = [int(labels[y, x]) for y, x in zip(ys, xs)]

    hit_labels = {lbl for lbl in point_comp if lbl > 0}
    n_hit = len(hit_labels)
    n_spurious = int(n_labels - 1 - n_hit)   # 去掉背景 label=0

    comps = {}
    for lbl in hit_labels:
        mask = (labels == lbl)
        ys_c, xs_c = np.where(mask)
        bbox = (int(xs_c.min()), int(ys_c.min()),
                int(xs_c.max()) + 1, int(ys_c.max()) + 1)
        comps[lbl] = dict(mask=mask, bbox=bbox)

    return dict(bin_mask=bin_mask, n_points=n_points, n_hit=n_hit,
                n_spurious=n_spurious, point_comp=point_comp, comps=comps)


def spurious_ratio(n_spurious, n_points):
    """虚警比：不过点组件数 / max(1, 点数)。

    口径沿用 PAL 的 alarm_point_ration（原先的
    (预测组件数 - 过点组件数) / 点数 > alarm_point_ration 判为"检测不好"），
    只是分母固定为点数、不再随预测组件数变化，语义更稳。
    """
    return n_spurious / max(1, n_points)


# =============================================================================
# 3. CLIP 图文相似度 K
# =============================================================================
class CIPatchScorer:
    """计算连通域局部窗与文本嵌入的余弦相似度均值 K。

    CLIP 图像塔由 `ClipTextEmbedder(keep_image_tower=True)` 持有并共享，
    文本嵌入 E 直接复用 TATE 那一份（保证"增强"与"筛选"看到的是同一个先验）。

    Args:
        clip_model: clip.load(...) 返回的模型（已 eval、requires_grad=False）。
        text_features: [N, 512] 文本嵌入（CPU/GPU 均可，float32）。
        device: torch device。
        side_ratio: 局部窗边长 = side_ratio * max(bbox 宽, 高)。
        min_side / max_side: 局部窗边长上下限（像素）。
        chunk: 一次送 CLIP 的 patch 数。
    """

    def __init__(self, clip_model, text_features, device,
                 side_ratio=1.5, min_side=16, max_side=64, chunk=128):
        if clip_model is None:
            raise ValueError(
                'CIPatchScorer 需要 CLIP 图像塔。请用 '
                'ClipTextEmbedder(..., keep_image_tower=True) 创建嵌入器。')
        self.clip = clip_model
        self.device = device
        self.side_ratio = side_ratio
        self.min_side = min_side
        self.max_side = max_side
        self.chunk = chunk

        text = text_features.detach().float().to(device)
        self.text_features = text / text.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        # CLIP 权重是 fp16 的，输入张量要跟着走
        try:
            self._dtype = next(clip_model.parameters()).dtype
        except StopIteration:
            self._dtype = torch.float32

    # ------------------------------------------------------------------ #
    def _crop(self, gray, bbox):
        """以 bbox 中心取正方形局部窗并预处理成 CLIP 输入 [3,224,224]。"""
        H, W = gray.shape
        x0, y0, x1, y1 = bbox
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        side = max(self.side_ratio * max(x1 - x0, y1 - y0), float(self.min_side))
        side = min(side, float(self.max_side))
        half = int(round(side / 2.0))
        half = max(half, 1)

        # 贴边时窗口会被截短，只影响窗口大小不影响中心（红外目标小，可接受）
        xs = max(0, int(round(cx)) - half)
        xe = min(W, int(round(cx)) + half)
        ys = max(0, int(round(cy)) - half)
        ye = min(H, int(round(cy)) + half)
        if xe <= xs:   # 退化保护
            xs, xe = max(0, int(cx) - 1), min(W, int(cx) + 1)
        if ye <= ys:
            ys, ye = max(0, int(cy) - 1), min(H, int(cy) + 1)

        patch = gray[ys:ye, xs:xe]
        patch = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_LINEAR)
        rgb = np.repeat(patch[:, :, None].astype(np.float32) / 255.0, 3, axis=2)
        rgb = (rgb - _CLIP_MEAN) / _CLIP_STD
        return rgb.transpose(2, 0, 1)

    # ------------------------------------------------------------------ #
    def score_components(self, gray, bboxes):
        """批量算 K。

        Args:
            gray: [H,W] uint8，原始灰度图（未归一化，直接来自磁盘）。
            bboxes: list[(x0,y0,x1,y1)]。
        Returns:
            np.ndarray [len(bboxes)]，每条是与 N 个 prompt 的余弦相似度均值。
        """
        if len(bboxes) == 0:
            return np.zeros(0, dtype=np.float32)

        patches = np.stack([self._crop(gray, b) for b in bboxes], axis=0)
        sims = []
        with torch.no_grad():
            for s in range(0, len(patches), self.chunk):
                t = torch.from_numpy(patches[s:s + self.chunk]).to(self.device)
                if self._dtype == torch.float16:
                    t = t.half()
                feat = self.clip.encode_image(t).float()
                feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                # [B, N] -> 对 prompt 维取均值（同 SeViL）
                sims.append((feat @ self.text_features.t()).mean(dim=1).cpu().numpy())
        return np.concatenate(sims, axis=0)


# =============================================================================
# 4. 图级均值 CI
# =============================================================================
def sample_mean_ci(comp_info, gray, pred_prob, scorer):
    """算一张图的 mean_ci 与逐点 CI。

    Args:
        comp_info: `extract_components` 的返回值。
        gray: [H,W] uint8 原始灰度图。
        pred_prob: [H,W] float32 概率图（S 用）。
        scorer: `CIPatchScorer`。
    Returns:
        dict(mean_ci=float, ci_per_point=list[float], S=list[float], K=list[float])
        未匹配到任何组件的点，其 CI 记 0（= 漏点惩罚，见文件头说明）。
    """
    labels = sorted(comp_info['comps'].keys())
    if len(labels) == 0:
        # 一个过点组件都没有：所有点都漏，mean_ci=0
        return dict(mean_ci=0.0, ci_per_point=[0.0] * comp_info['n_points'],
                    S=[], K=[])

    bboxes = [comp_info['comps'][lbl]['bbox'] for lbl in labels]
    Ks = scorer.score_components(gray, bboxes)

    ci_of_label, S_list, K_list = {}, [], []
    for lbl, K in zip(labels, Ks):
        S = component_confidence(pred_prob, comp_info['comps'][lbl]['mask'])
        ci_of_label[lbl] = credibility_index(S, float(K))
        S_list.append(S)
        K_list.append(float(K))

    ci_per_point = [ci_of_label.get(lbl, 0.0) for lbl in comp_info['point_comp']]
    if len(ci_per_point) == 0:
        mean_ci = 0.0
    else:
        mean_ci = float(np.mean(ci_per_point))

    return dict(mean_ci=mean_ci, ci_per_point=ci_per_point, S=S_list, K=K_list)


# =============================================================================
# 5. GMM 动态阈值（SeViL gmm_policy 的 policy='high' 分支，逐行对齐）
# =============================================================================
def gmm_threshold(values, min_samples=4):
    """对一批 CI 拟合 2 分量 GMM，返回高簇密度峰值处的 CI 作为阈值。

    判决逻辑与 SeViL 的 `gmm_policy(policy='high')` 逐行一致：
        1. 找出每个点的混合 log 密度；
        2. 屏蔽低簇后取 argmax，即"高簇里密度最大的那个样本"，
           其 CI 即阈值（效果上卡掉低簇 + 高簇下半边）。
        高簇由 `gmm.means_` 判定（均值大的那个），不依赖标签编号。

    ⚠️ **与 SeViL 唯一且必要的偏离：不传 `means_init/precisions_init`**。
    SeViL 用的是 `precisions_init=[[[1.0]]]`（先验方差 = 1.0），而 CI 的实际
    尺度只有 0.1~0.9，这个先验比数据方差大一个量级，实测会让两个分量几乎
    完全重叠、退化成"求众数"：阈值落到分布中心附近，每轮固定放行约 50%
    （零堆积时 67%），完全起不到节流作用 —— 而这正是本项目要解决的问题。
    改用 sklearn 默认初始化后，分量才能真正分离，阈值 = 高簇中心，
    双峰分布下放行 ~25%，才能真正替代 PAL 的人工阈值。

    Args:
        values: list/np.ndarray，样本级（图级）CI。
        min_samples: 少于该数量时返回 None，由调用方回退。
    Returns:
        float 或 None
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    if len(v) < min_samples:
        return None

    with warnings.catch_warnings():
        # 全部数值相同时 sklearn 会退化成单簇并告警，这里已做退化保护
        warnings.simplefilter('ignore')
        gmm = GaussianMixture(2)     # 默认初始化（kmeans），见上方说明
        gmm.fit(v)

    labels = gmm.predict(v)                      # 0/1
    high = int(np.argmax(gmm.means_.ravel()))    # 自行判定高簇，不依赖编号
    assign = (labels == high)
    if not assign.any():
        return float(np.max(v))                  # 退化保护：全在一个簇

    llh = gmm.score_samples(v).copy()
    llh[~assign] = -np.inf                       # 低簇不参与
    idx = int(np.argmax(llh))                    # 高簇密度峰值
    pos = assign & (v.ravel() >= v.ravel()[idx])
    if not pos.any():                            # 退化保护（全 -inf / 非有限值）
        return float(v[assign].min())
    return float(v[pos].min())


def fallback_threshold(values, mode='median'):
    """GMM 样本不足时的回退阈值。

    SeViL 回退固定 0.3，但 CI 在红外域的量纲与之不同，固定值没有意义；
    这里回退"池中位数"（等价放行约一半），语义稳定。
    """
    if len(values) == 0:
        return 0.0
    v = np.asarray(values, dtype=np.float64)
    if mode == 'fixed':
        return 0.3
    return float(np.median(v))


# =============================================================================
# 6. 数量闸：本轮最多放行 k 张
# =============================================================================
def admit_ratio(epoch, num_epochs, ratio_init=0.05,
                enh_start_ratio=0.2, enh_end_ratio=0.8):
    """当前 epoch 的"最多放行比例"（占困难池），随 enhancement 阶段线性升到 1.0。

    为什么需要它：GMM 阈值只管"谁比谁好"，不管"这轮一共能放几个"。
    若某轮 CI 分布是单峰（模型普遍自信 / 普遍糟糕），任何阈值都只能砍掉约一半，
    仍会出现"一次进太多"。数量闸把每轮新增量钉在池子的固定比例上，
    斜率与 PAL 的 `lose_point_ratio` 爬坡一致：
        起点 ratio_init（默认 0.05，即早期每轮最多放行 5% 的困难池），
        到 enhancement 末尾升到 1.0（= 不限制，完全交给 theta）。
    这样既压住早期震荡，又保证困难池在 refinement 阶段前能排空
    （否则后 20% 的 epoch 会只用很少的样本训练）。

    Args:
        epoch: 当前 epoch（0-based）。
        num_epochs: 总 epoch 数。
        ratio_init: 起点比例。
        enh_start_ratio / enh_end_ratio: enhancement 阶段在总 epoch 中的位置。
    Returns:
        float in [ratio_init, 1.0]
    """
    span = max(1e-8, num_epochs * (enh_end_ratio - enh_start_ratio))
    prog = (epoch - int(num_epochs * enh_start_ratio) + 1) / span
    prog = min(max(prog, 0.0), 1.0)
    return float(min(1.0, ratio_init + prog * (1.0 - ratio_init)))
