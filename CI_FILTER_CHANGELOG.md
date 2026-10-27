# 改动记录：用 CI + GMM 门控替换 PAL 的困难样本筛选（TAPF 风格）

> 日期：2026-10
> 动机：原 PAL 的困难样本筛选（COU）每轮放行过多 → 伪标签池突变 → 训练震荡。
> 目标：把 SeViL 的 TAPF 机制（CI + GMM 动态阈值）改造成"单帧红外 + 点监督"可用的形式，
> 作为困难样本筛选的**唯一判据**（默认启用）。

---

## 1. 改了什么（按文件）

| 文件 | 状态 | 内容 |
|---|---|---|
| `PAL-main/components/credibility.py` | **新增** | CI 门控全部实现：组件提取/过点匹配、S、K(CLIP)、图级均值 CI、GMM 阈值、数量闸比例 |
| `PAL-main/components/text_prior.py` | 改 | `ClipTextEmbedder` 增加 `keep_image_tower` 参数与 `get_image_tower()`：CI 与 TATE 共用同一份 CLIP 与文本嵌入 E；默认 `False` 时行为与原来完全一致 |
| `PAL-main/train_model.py` | 改 | enhancement 阶段的困难样本筛选改为 CI 门控（`USE_CI_FILTER=True`，默认）；原 PAL 判据保留在 `else` 分支供 A/B |
| `PAL-main/tools/test_ci_filter.py` | **新增** | 自检脚本（不依赖数据集/CLIP），验证组件提取、S/CI、漏点惩罚、GMM 阈值与放行率 |

**未改动**：`utilts.py` 的所有原函数（`nc_pred_mask` / `deal_pred_mask_and_true_point_in` /
`deal_gen_mask_error_aera` / `hard_sample_in`）一行未删。其中
`deal_gen_mask_error_aera` 与 `hard_sample_in` **继续被复用**——前者恰好实现了
"只保留过点连通域 ∪ 点标注"，与 CI 门控的精炼语义一致。

---

## 2. 算法规则（现行为）

每 `clear_epoch_gap=5` 个 epoch、在 enhancement 阶段（`0.2N ~ 0.8N`）执行一次：

```
# 阶段 A：纯几何（不消耗 CLIP）
对每张困难样本：
    prob = sigmoid(model(x))；bin = prob > 0.5
    连通域标记；按点标注把连通域分为"过点组件"与"不过点组件（疑似虚警）"
    if 点数 == 0: 直接按原逻辑进简单池（无点可学）
    if 虚警数 / max(1, 点数) > alarm_point_ration(=5): 本轮不学，留在困难池

# 阶段 B：CI（只有通过预筛的样本才用 CLIP 图像塔）
S = 0.5*max(prob[comp]) + 0.5*mean(prob[comp])
K = mean_i cos( CLIP_img(局部窗), E_i )        # E 与 TATE 共用
CI = 2SK/(S+K);  未匹配到组件的点记 CI = 0
mean_ci(图) = mean(该图各点的 CI)               # = 图级统计单元

# 阶段 C：GMM 动态阈值（对图级均值拟合，一轮一个 theta，跨轮 EMA 平滑）
theta_gmm = GMM(2) 高簇密度峰值处的 CI     # SeViL gmm_policy(policy='high')
theta     = 0.5*theta_prev + 0.5*theta_gmm  # 防抖；样本<4 时回退池中位数
# 特殊：若 theta_gmm <= 0（所有点都没检出 / 池为空）→ 本轮 theta=inf，全不放行，
#       且不把这个 0 写进 EMA（否则阈值会被永久拉成 0）

# 阶段 D：放行（theta 定"够不够格"，k 定"这轮最多进几个"）
passed   = [图 for 图 if mean_ci >= theta]
k        = max(1, round(admit_ratio(epoch) * 困难池总数))   # 比例 0.05 → 1.0 线性爬坡
accepted = passed 按 mean_ci 从高到低取前 k 个              # 差额留在困难池下一轮再判
# 随后：deal_gen_mask_error_aera 只保留过点连通域并融合点标注 → hard_sample_in 搬文件
```

关键设计点（与 SeViL 的差异，均为适配"单帧红外 + 点监督 + 无检测头"）：

1. **S**：红外分割网没有 bbox score，用概率图在连通域内的 `0.5*max + 0.5*mean`。
2. **K**：红外目标只有几个像素，按 bbox 裁剪再放大等于插值噪声；改为
   "以连通域中心取正方形局部窗"（边长 `1.5×bbox长边`，夹在 `[16, 64]` px），
   再 resize 224。像素取自**磁盘原图**（不做 albumentations 归一化污染）。
3. **统计单元从"实例级"改为"图级均值"**：PAL 需要"整张样本进不进简单池"的判决。
   `mean_ci` 以该图**点数**为分母，未匹配点记 0 —— 于是"漏点"会自动把均值拉低，
   这正好替代了 PAL 原 `lose_point_ratio` 的渐进召回率爬坡，**不再需要召回率条件**。
4. **阈值只在图级均值上拟合**（不是组件级）：均值方差 ≈ σ²/p，与组件级分布不同源，
   混用会让阈值系统性偏严。

### 2.1 ⚠️ 与 SeViL 的一处**刻意偏离**：GMM 不传 `precisions_init`

SeViL 用 `means_init=[[min],[max]] + weights=0.5 + precisions_init=1.0`。
CI 的尺度只有 0.1~0.9，**先验方差 1.0 比数据方差大一个量级**，实测会让两个分量
几乎完全重叠、退化成"求众数"。合成数据实测（120~600 点）：

| 数据形态 | SeViL 原初始化 | sklearn 默认初始化（本实现） |
|---|---|---|
| 均匀双峰 0.1~0.2 / 0.6~0.7 | θ=0.600，放行 **50%** | θ=0.649，放行 **26%** |
| 高斯双峰 0.15 / 0.55 | θ=0.402，放行 **50%** | θ=0.550，放行 **25%** |
| 零堆积 + 单峰（早期常见） | θ=0.350，放行 **67%** | θ=0.503，放行 **34%** |
| 单峰 | θ≈中位数，放行 50% | θ≈中位数，放行 48% |

结论：原初始化下每轮固定放行约一半，**根本起不到节流作用**，与本项目动机冲突；
故改用默认初始化（分量能真正分离），判决逻辑（高簇屏蔽低簇 → 取密度峰值）保持与 SeViL 一致。
`tools/test_ci_filter.py` 里有回归断言：双峰数据放行率必须落在 15%~35%。

- **已知残余风险**：若某轮 CI 分布是**单峰**（模型普遍自信/普遍糟糕），任何阈值都只能
  砍掉一半左右。对这种情况的兜底是**数量闸**（见下），它不依赖分布形状。

---

## 2.2 数量闸 k（`CI_USE_QUOTA=True`）

θ 只管“谁比谁好”，不管“这轮一共能放几个”。因此再加一道数量闸：

```
ratio = 0.05 → 1.0，在 enhancement 阶段内线性爬坡（斜率同 PAL 的 lose_point_ratio）
k     = max(1, round(ratio * 当前困难池总数))      # 基数在搬文件前取
accepted = 过 theta 的样本按 mean_ci 从高到低取前 k 个
```

- 早期 ratio=0.05：2000 张的池子一轮最多放 100 张，直接压住“一次进太多”；
- 末尾 ratio→1.0：数量闸自动退化成不限制，**完全交给 θ 把关**；
- 保证困难池在 refinement 阶段前能排空（否则后 20% epoch 只能用很少样本训练）；
- 被 k 卡掉的样本留在困难池，下一轮重新参与（不会被永久丢弃）。
- `CI_USE_QUOTA=False` 可关闭（仅用于验证数量闸本身的贡献）。

---

## 3. 配置项（`train_model.py` 顶部）

```python
USE_CI_FILTER      = True     # 唯一的回退开关：False = 原版 PAL 门控（A/B 对照）
CI_EMA_BETA        = 0.5      # 阈值跨轮 EMA 平滑
CI_MIN_POOL        = 4        # 少于此样本数不做 GMM，回退池中位数
CI_FALLBACK        = 'median' # 回退口径：'median' | 'fixed'(0.3)
CI_USE_QUOTA       = True     # 数量闸开关（False = 只靠 theta）
CI_ADMIT_RATIO_INIT = 0.05    # 数量闸起点比例（占困难池），随 enhancement 线性升到 1.0
CI_CROP_SIDE_RATIO = 1.5      # CLIP 局部窗边长系数
CI_CROP_MIN/MAX    = 16 / 64  # 局部窗边长范围（像素）
alarm_point_ration = 5        # 复用原参数作为虚警预筛上限（虚警数/点数）
```

`TEXT_PRIOR_RANDOM=True` 时，**CI 门控也使用随机文本嵌入**——这是
"文本是否真的携带信息"消融的正确语义（若只随机化 TATE 而门控仍用真文本，消融不成立）。

---

## 4. 新增的日志 / 产物（`work_dirs/<MODEL_NAME>/`）

- 主日志追加 `[CI-Filter] ...` 行：候选数、虚警预筛挡下数、`theta_gmm`、`theta_ema`、
  过 theta 数、`k`、被 k 卡掉数、放行数、放行比例、`mean_ci` 的 p10/p50/p90、`avgS`、`avgK`。
- 主日志额外两行：**单点图 / 多点图**分组的样本数与放行数
  （单点图的 `mean_ci` 是单个组件值、噪声大；分组才能判断多点图是否被系统性卡住）。
- `ci_theta.json`：`{epoch, theta_gmm, theta_ema}`，供 `--resume1` 恢复阈值 EMA。
- `ci_log/ci_round_epochXXXX.jsonl`：逐样本 `mean_ci / 每点 CI / S / K / 点数 / 虚警数 / 是否放行`。

---

## 5. 验证与回归

```bash
cd PAL-main
python tools/test_ci_filter.py     # 纯 numpy 部分，无需 GPU/CLIP/数据
```

已验证项：组件提取与过点匹配、虚警比口径、S 与 CI 公式、"漏点惩罚"（未匹配点 CI=0
使 `mean_ci` 减半）、GMM 阈值落点与放行率、少量样本回退、退化输入保护、
数量闸比例插值（起点 0.05 → 末尾 1.0）。

> 本容器无 torch，`components/credibility.py` 的 CLIP 分支（`CIPatchScorer`）只能在
> 训练机上验证：`python tools/test_ci_filter.py --with_clip --clip_path <ViT-B-32.pt>`。

---

## 6. 训练前必读

- **伪标签池会被原地改写**（见 `AGENTS.md §5.1`）。`USE_CI_FILTER` 的 A/B 必须各自使用
  独立的初始池（独立 `MODEL_NAME` 或拷贝 `dataset/`），否则两组的池子演化不可比。
- 首个更新轮次（epoch = 0.2N+5）会打印 `[CI-Filter]` 一行。做一次快速判定：
  - `theta` 是否明显高于 `mean_ci p50`（否则阈值没有区分力）；
  - `avgK` 是否几乎恒定（若 K 方差极小，说明 CLIP 对红外小目标不敏感，CI 退化为只看 S）；
  - 放行数是否远小于 hard 池（目标：明显小于原 PAL 的一轮几百张）。
- 若 `avgK` 退化，优先调整 `CI_CROP_MIN/MAX`（局部窗太小 → 无上下文；太大 → 目标占比过低）。
