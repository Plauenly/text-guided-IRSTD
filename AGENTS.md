# AGENTS.md —— PAL+TATE

> 本文件面向后续在本目录工作的 AI/研究者。
> **先读本文件，再动代码。** 本目录是原版 PAL 的一个**实验分支**，用于验证
> "把 SeViL 的文本先验（TATE）引入点监督单帧红外小目标检测" 是否有效。

---

## 0. 一句话定位

| 项 | 内容 |
|---|---|
| 基础工作 | **PAL**（ICCV 2025，*From Easy to Hard: Progressive Active Learning*，单点监督单帧 SIRST） |
| 引入模块 | **TATE**（SeViL, AAAI 2026，*Text-guided Adaptive Target Enhancing*，文本先验特征增强） |
| 任务 | 单帧红外小目标分割（点监督：coarse point / centroid point） |
| 当前阶段 | **快速验证**：只做特征增强，只支持 **MSDA** 一个模型 |
| 数据集 | SIRST3（默认），另有 IRSTD_1K / NUDT_SIRST / SIRST_1_1 |
| 框架 | **纯 PyTorch 单文件训练循环**，与 mmdet 无关 |

> ⚠️ 与本 workspace 下的另外两个项目（`UCOD-MKD` 无监督 COD、`SeViL` 半监督移动红外）
> **监督设定与框架完全不同，不要混用术语或代码。**

---

## 1. 目录结构

```
PAL+TATE/                      ← git 仓库根目录
├── AGENTS.md                  ← 本文件
├── .gitignore
├── pal_tate_full.patch        ← 完整补丁（原版 PAL → 本仓库）
├── pal_tate_dtype_fix.patch   ← v1 补丁（DWT dtype 修复 + test 脚本 --clip_path）
├── pal_tate_step2.patch       ← step2 补丁（test_model 支持文本先验 + AMP dtype 小修）
├── CI_FILTER_CHANGELOG.md     ★★ CI 门控改动记录（用 CI+GMM 替换 PAL 困难样本筛选）
└── PAL-main/                  ← 实际代码
    ├── train_model.py         ★ 训练主循环 + 主动学习调度 + 文本先验接入
    ├── test_model.py          ★ 测试（已支持文本先验）
    ├── utilts.py              ★ EPG 初始伪标签生成 / 难易样本划分 / 伪标签精炼
    ├── components/
    │   ├── text_prior.py      ★★ 文本先验模块（TATE）—— 本分支核心新增
    │   ├── text_prompts.py    ★★ 单帧红外 prompt 列表
    │   ├── credibility.py     ★★ CI 门控（S/K/CI + GMM 阈值）—— 替换 PAL 困难样本筛选
    │   ├── dataset_final_edge_copy_paste_final_2_img_path.py  数据集（返回 img/mask/edge）
    │   ├── utils_all_edge_copy_paste_final_2_img_path.py      DataLoader 封装
    │   ├── metric_new_crop.py 指标（IoU / nIoU / Pd / Fa）
    │   ├── edges.py, gardient.py, cal_mean_std.py
    ├── loss/Edge_loss.py      ★ EEDM 损失（edgeSCE_loss）
    ├── model/{MSDA,ACM,ALC,MLCL,ALCL,DNA,GGL,UIU}/*_no_sigmoid.py   8 个可选网络
    ├── tools/
    │   ├── test_text_prior.py ★ 文本先验自检脚本（不依赖数据集）
    │   ├── test_ci_filter.py  ★ CI 门控自检脚本（纯 numpy，不依赖 GPU/CLIP）
    │   ├── coarse_anno.m, centroid_anno.m       点标注生成（MATLAB）
    ├── dataset/               【.gitignore 忽略】790MB 数据 + 伪标签工作池
    ├── imgs/                  【.gitignore 忽略】png 可视化
    ├── work_dirs/             【⚠️ 目前未完全忽略】训练日志 + 权重
    └── TEXT_PRIOR_README.md   ★ 文本先验的安装/运行/实验说明
```

★ = 阅读重点；★★ = 本分支的核心。

---

## 2. 相对原版 PAL 的改动清单

**未改动任何模型的网络结构逻辑**（文本先验用 forward hook 零侵入注入）。

| 文件 | 状态 | 改动内容 |
|---|---|---|
| `components/text_prompts.py` | **新增** | 单帧红外 prompt 10 条（SeViL 13 条裁剪而来） |
| `components/text_prior.py` | **新增** | `ClipTextEmbedder` + `TextPriorAttention`(TATE) + `attach_text_prior()` |
| `components/credibility.py` | **新增** | CI 门控：组件提取/过点匹配、S、K(CLIP)、图级均值 CI、GMM 动态阈值 |
| `CI_FILTER_CHANGELOG.md` | **新增** | CI 门控的完整改动记录、公式、校准实验与日志说明 |
| `tools/test_ci_filter.py` | **新增** | CI 门控自检（组件/S/CI/漏点惩罚/GMM 阈值与放行率） |
| `tools/test_text_prior.py` | **新增** | 自检脚本（模块 / MSDA 接入 / CLIP 编码） |
| `PAL-main/TEXT_PRIOR_README.md` | **新增** | 使用说明 |
| `train_model.py` | 改 | 顶部 7 个开关；模型构造后接入文本先验 |
| `train_model.py` | 改 | **困难样本筛选改为 CI 门控（默认开启）**，原 PAL 判据保留在 `else` 分支；新增 `[CI-Filter]` 日志与 `ci_log/` |
| `components/text_prior.py` | 改 | `ClipTextEmbedder` 增加 `keep_image_tower`：CI 门控复用同一份 CLIP 与文本嵌入 E |
| `test_model.py` | 改 | 同样接入文本先验；`load_state_dict(strict=False)` + 不匹配提示 |
| `model/MSDA/MSDA_no_sigmoid.py` | 改 | **修原版 bug**：5 处 DWT 反向硬编码 fp16 → 跟随 `dx.dtype` |

原版 PAL 已有的问题（本分支已处理）：
- `epoch = 100` 硬编码 → **已注释**（`train_model.py`）
- `val_fn` 里 `eval_losses` 未定义 → **已修**
- `check_path` 强制退出 → **已注释**（改为复用已有工作池）

---

## 3. 文本先验（TATE）设计

### 3.1 数据流

```
输入 256×256×3
  → MSDA_No_Sigmoid
      └── stage (Stage)            # 5 级编码器
            ├── resnet1..4          # 主路径不降采样
            ├── layer1_4..4_4       # 侧路池化（/16, /8, /4, /2）
            └── layer5_4            # ★ 拼接后 128ch → 64ch，输出 [B,64,16,16]
                                        ↑ 这里被 forward hook 注入文本注意力
      → mlcl5..1 → sbam4..1 → self.layer → 单通道 logits
```

### 3.2 注入点（关键）

```python
MODEL_TEXT_PRIOR_CFG = {
    'MSDA': dict(layer_name='stage.layer5_4', feat_dim=64),
}
```

- 256 输入时，bottleneck 是 **16×16×64**（已实测确认）。
- 通过 `target.register_forward_hook(_hook)` 注入，**不修改模型源码**。
- 文本模块用 `model.add_module('text_prior', ...)` 注册，因此
  `optimizer = AdamW(model.parameters())` 自动包含其参数。

### 3.3 公式（对应 SeViL Eq.1–3）

```
E  = CLIP_text(prompts)                  # [N,512]，N=10
Ep = expand( mean_i( f_mlp(E_i) ) )      # f_mlp: 512→256→feat_dim 的 MLP
G  = Softmax( f_c( M ⊙ Ep ) )            # f_c: 1×1 conv + 空间 softmax
M_E = f_c( G ⊙ M + M )                   # 残差加权后再 1×1 conv
```

### 3.4 三个刻意的设计决策（**改之前先想清楚**）

1. **通道自适应**：SeViL 文本与视觉都是 256 维可直接 `expand_as`；
   PAL 的 bottleneck 只有 64 维，故 `mlp_vis` 输出维度 = `feat_dim`。
2. **文本嵌入缓存为 buffer**：SeViL 每次 forward 都重新 `encode_text`（浪费）；
   本实现只在初始化时编码一次，随后**立即释放 CLIP**，训练零额外开销。
3. **残差门控 `alpha`（初值 0）**：训练初期该模块等价**恒等映射**，
   不会破坏 baseline 收敛，由网络自己学习"要不要用文本先验"。
   这是在新域（CLIP 没见过红外小目标）下的保险措施。

### 3.5 文本 prompt

- 位置：`components/text_prompts.py`，当前 `SINGLE_FRAME_IR_PROMPTS` 共 **10 条**。
- 来源：SeViL 的 13 条，**删掉 3 条多帧/运动专属**（文件头注释里写明了删了哪几条）。
- 另有 `_MINIMAL`（4 条）与 `_TINY`（2 条）备用，用于 prompt 数量消融。

---

## 3A. CI 门控：替换 PAL 的“困难样本筛选”（默认开启）

> 背景：PAL 原 COU 判据（`utilts.nc_pred_mask`：点位漏检率 + 虚警比）每轮放行过多，
> 伪标签池突变 → 训练震荡。改用 SeViL TAPF 的 `CI = 2SK/(S+K)` + GMM 动态阈值。
> 完整记录（含校准实验数据）见 `CI_FILTER_CHANGELOG.md`；公式细节见
> `PAL-main/components/credibility.py` 文件头。

### 三步（每 `clear_epoch_gap=5` 轮一次，仅 enhancement 阶段）

1. **阶段 A（纯几何，不耗 CLIP）**：连通域 → 按点标注分“过点组件 / 不过点组件”；
   `虚警数/max(1,点数) > alarm_point_ration(5)` → 本轮不学，留在困难池。
2. **阶段 B（CI，只对通过预筛的样本）**：
   `S = 0.5*max(prob)+0.5*mean(prob)`（组件内）
   `K = mean_i cos(CLIP_img(局部窗), E_i)`（E 与 TATE 共用）
   `CI = 2SK/(S+K)`；未匹配点记 0；
   **图级均值** `mean_ci = mean(各点 CI)`。
3. **阶段 C+D**：对全体 `mean_ci` 拟合 GMM(2)，取高簇密度峰值作 `theta`（跨轮 EMA 平滑，
   `<4` 样本回退池中位数）；**唯一判据** `mean_ci >= theta` → 进简单池。
   随后复用 `deal_gen_mask_error_aera`（只留过点连通域 ∪ 点标注）+ `hard_sample_in`。

### 必须知道的四个点

1. **GMM 不传 `precisions_init`**（与 SeViL 的唯一刻意偏离）。CI 尺度 0.1~0.9，
   SeViL 的先验方差 1.0 会让两分量重叠、退化成求众数 → 每轮固定放行 ~50%，
   反而无法节流；默认初始化下双峰数据放行 ~25%。`tools/test_ci_filter.py` 有回归断言。
2. **`mean_ci` 自带漏点惩罚**（未匹配点 CI=0），所以**不再需要** `lose_point_ratio` 爬坡；
   但 enhancement 里仍保留该变量的计算（旧分支会用），CI 分支不依赖它。
3. **单峰风险**：若某轮 CI 分布单峰（普遍自信/普遍糟糕），θ 只能砍一半——
   这就是数量闸 k 存在的原因（它不依赖分布形状）。早期 `ratio=0.05` 直接压住入场量，
   末尾 `ratio→1.0` 退化为不限制、完全交给 θ。`CI_USE_QUOTA=False` 可关（仅用于消融）。
4. **`TEXT_PRIOR_RANDOM=True` 时 CI 也用随机文本**——这是“文本是否有信息量”消融的正确语义。

### 上线前的快速判定（首个 `[CI-Filter]` 日志行）

- `theta` 是否明显高于 `mean_ci p50`；
- `avgK` 是否几乎恒定（若是，CI 退化为只看 S，需调 `CI_CROP_MIN/MAX`）；
- 放行数是否**明显小于**原 PAL 一轮几百张。

日志：主 txt 里的 `[CI-Filter]` 行 + `ci_theta.json` + `ci_log/ci_round_epochXXXX.jsonl`
（逐样本 `mean_ci / 每点 CI / S / K / 点数 / 虚警数 / 是否放行`）。

---

## 3B. 文本适配器 + ACM + 困难池排空（默认开启）

> 目标：让"文本"真正参与（而非只有一个标量门控），并把 CI 门控后期锁死的困难样本收回训练。
> 公式、参数与验证方式见 `CI_FILTER_CHANGELOG.md §7`；实现见
> `PAL-main/components/cross_modal.py`。

### 三个模块

1. **TextAdapter**（`Ea = E + gate·MLP(E)`，gate 零初始化）：在缓存的 CLIP 文本嵌入上微调，
   代价 ~0.5M 参数 / 每步 5 MMACs。TATE 与 ACM 共用同一实例（`model.text_adapter`，
   只注册一次，state_dict 不重复）。
2. **ACM（512 维直连）**：`prefusion(128ch,16x16) --1x1conv--> V(512)`，
   `Q = normalize(Ea 元素级掩码 15%)`，`out = softmax(Q·Vᵀ/√512)·V`，
   `loss = MSE(out, normalize(Ea))` **只在被 mask 的元素上**。
   - **只对 `attn·V` 求损失**（无残差旁路），否则 α→0 会退化成纯文本任务、视觉侧拿不到梯度；
   - `ACM_GRAD_TO_BACKBONE=True` 时梯度回传主干（让特征"读得出"文本语义）；
   - 捕获点是 `stage.layer5_4` 的**输入**（融合前），用 `forward_pre_hook`。
3. **排空**（`CI_DRAIN`）：refinement 阶段（>0.8N）按最近一轮 `mean_ci` 降序、
   以 0.2→1.0 的占比逐步把剩余困难样本收进简单池，初始伪标签 = 点标注，随后由 FIU 演化。
   这补齐了 PAL 原版 `lose_point_ratio→1.0`"后期全收"的语义。

### 关键开关（`train_model.py` 顶部）

```python
USE_TEXT_ADAPTER=True  TEXT_ADAPTER_HIDDEN=512  TEXT_ADAPTER_GATE_INIT=0.0
USE_ACM=True  ACM_LAMBDA=0.1  ACM_WARMUP_EPOCHS=10  ACM_MASK_RATIO=0.15
ACM_GRAD_TO_BACKBONE=True
CI_DRAIN=True  CI_DRAIN_START_RATIO=0.8  CI_DRAIN_END_RATIO=0.95  CI_DRAIN_START_FRAC=0.2
```

### 必须看的三条日志

- `acm_loss / masked_mse / kept_mse`：`masked` 应显著低于 `kept` 且都下降；
- `adapter_gate` 与 `tate_alpha`：判断模型"到底用了多少文本"的直接证据；
- `[Drain]` 行：剩余池应单调下降并在 0.95N 前归零。

### 注意

- **筛选用的 K 跟随适配器**（`CIPatchScorer.set_text_features`），所以 K 不再是独立标尺；
  做 `TEXT_PRIOR_RANDOM` 消融时必须同时打开，否则分不清"适配器拟合"与"文本语义"。
- 排空不区分"不可学样本"（按要求）：A 类退化样本也会进来，初始标签只有点，
  不会把大块伪前景带进训练。

---

## 3C. 尺寸统一约定：**原图坐标 + 只 pad 不 resize**（commit cfbcdf7 起）

一句话：**除了训练时对 image+mask 做同一个窗口的 256 随机裁剪，其余环节（val / test / FIU /
伪标签落盘 / CI 筛选）全部在原图坐标上做，只把右/下 pad 到 32 的倍数，绝不 resize。**

- 训练：`random_crop(image, mask, 256)` 一次调用共用同一 `h_start/w_start`；图小于 256 先右下 pad。
  `train_transform` 里**没有** crop/resize，只有翻转/旋转/亮度/噪声 + Normalize。
- val/test/FIU：`pad_to_multiple(...,32)` 后返回原图 `h,w`；预测裁回 `[:, :, :h, :w]`，指标与
  `update_gt_update_degen_corr` 都在原图坐标；伪标签按原图坐标写盘。
- 尺寸不一 ⇒ **val/FIU 的 DataLoader 一律 `batch_size=1`**；只有 `h,w>1024` 才 `unfold/fold` 分块
  （`stride=patch_size`，当前三个数据集最大 512，不触发）。
- 兼容旧残留 256×256 伪标签：`match_mask_to_image()` / `utilts.py` 内 nearest resize 兜底并 `[warn]`；
  出现该警告说明池子没重生成，应删池重跑。
- ⚠️ 历史坑：`0b63f19`~`cfbcdf7` 之前训练用 random_crop(256)、val/FIU 用整图 resize 到 256，
  伪标签在 256 尺度演化后再放回原图 —— 几何不一致 + `33×33` 核相对覆盖面积漂移（33×33 方块成因之一）。
- 仍未统一的是**输入尺度上下文**：训练看 256 原生裁块（512² 图上可能裁空），FIU 看整图。

---

## 3C. 高斯 Focal Loss（默认替换 edgeSCE）

`LOSS_TYPE='gauss_focal'`，`LOSS_TYPE='edgeSCE'` 可一键回退。

- **热图**：`components/heatmap.py`，由**点标注**生成，σ=1.5（patch 像素）、半径 3σ、
  多目标取 max、峰值强制 1.0（否则 CornerNet 的 `pos==1` 判定会失效）。
- **损失**：`loss/gaussian_focal_loss.py`，target=**伪标签**，热图只做权重：
  核内权重 `1+(W-1)g`（W=4，点处最大）、核外 `(1-g)^γ`；`by_pos_neg` 归一化
  （量级 ≈0.12~0.17，与 edgeSCE 同档，ACM 的 λ 不用改）。
- **crop**：`PATCH_SIZE=320`、`CROP_FOV=0.8`（512 图缩放 0.781、256 图 1.25）。
  点的坐标用数学缩放重建，不对点图重采样。
- **几何增强**：伪标签与点堆成 2 通道 mask 一起送 albumentations
  （避免依赖 `masks=[...]` 关键字，版本兼容性更稳）。
- **日志**：`task / p_pos / p_neg / pos_px / pred_fg / empty`，含义见
  `CI_FILTER_CHANGELOG.md §8.4`。
- **自检**：`python tools/test_heatmap.py`（无 torch 可跑）。

⚠️ 注意：`target=pseudo` 意味着损失不会对抗伪标签膨胀（那是 `GF_TARGET='heatmap'` 的能力）。
方块膨胀仍由 FIU + CI 门控控制。

---

## 4. 运行方式

### 4.1 依赖

```bash
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git      # 或 pip install -e /workspace/SeViL/CLIP
```

### 4.2 CLIP 权重

- 默认 `clip.load('ViT-B/32')` 自动下载到 `~/.cache/clip/ViT-B-32.pt`。
- 手动下载：
  ```bash
  mkdir -p ~/.cache/clip
  wget -O ~/.cache/clip/ViT-B-32.pt \
    https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
  ```
- 或改 `train_model.py` 的 `TEXT_CLIP_MODEL = '/绝对路径/ViT-B-32.pt'`。

### 4.3 自检（**改完文本先验代码后必跑**）

```bash
cd PAL-main
python tools/test_text_prior.py                    # 不需要 CLIP 权重
python tools/test_text_prior.py --with_clip --clip_path /path/ViT-B-32.pt
python tools/test_ci_filter.py                     # CI 门控纯 numpy 自检（无 GPU/CLIP/数据）
python tools/test_ci_filter.py --with_clip --clip_path /path/ViT-B-32.pt   # 额外验证 CLIP 局部窗
```

### 4.4 训练 / 测试

```bash
cd PAL-main
python train_model.py                    # 从 cwd 读 dataset/，必须在 PAL-main 下运行
python train_model.py --resume1          # 从 enhancement 起点续训（只认第一阶段权重）
python test_model.py                     # 需先填 test_dir_name
```

### 4.5 关键开关（`train_model.py` 顶部）

```python
choose_model          = 'MSDA'        # 目前只有 MSDA 配了文本先验注入点
choose_dataset        = 'SIRST3'
choose_dataset_type   = 'masks_coarse' # 点监督；'mask'=全监督
NUM_EPOCHS            = 400
USE_TEXT_PRIOR        = True          # False => 等价于原版 PAL（baseline）
TEXT_PRIOR_RANDOM     = False         # True  => 随机文本嵌入（关键消融）
TEXT_CLIP_MODEL       = 'ViT-B/32'
TEXT_PRIOR_USE_GATE   = True
TEXT_PRIOR_GATE_INIT  = 0.0
os.environ['CUDA_VISIBLE_DEVICES'] = '2'   # ⚠️ 硬编码，换机器必改
```

---

## 5. 实验协议（**违反则结论无效**）

### 5.1 ⚠️ 伪标签池会被"原地修改"

工作池位于 `dataset/<数据集>/<MODEL_NAME>/`，训练中会被**就地改写**：

- `train/choose/mask/` 每个 epoch 被覆盖（伪标签自更新）
- `train/no_choose/` 的样本被搬走（`hard_sample_in` **边移动边删除**）

而初始池只在 `TRAIN_IMG_DIR` **不存在**时才生成。

**因此：baseline 与 TATE 必须从同一份初始池出发。**
推荐做法（改一行）：

```python
MODEL_NAME = choose_model + '__' + choose_dataset + '__' + choose_dataset_type + '_TATE'
```

初始池生成（`data_inital_make_add_points`）是**确定性的**（高斯+Canny，无随机），
所以删除后重生成能得到同一份初始池。

### 5.2 对照矩阵

| 实验 | 设置 |
|---|---|
| baseline | `USE_TEXT_PRIOR=False` |
| +TATE | 默认（真实文本） |
| +TATE(random) | `TEXT_PRIOR_RANDOM=True` —— **验证文本是否真有信息量的关键消融** |
| +TATE(no gate) | `TEXT_PRIOR_USE_GATE=False` |
| 原 PAL 筛选（A/B） | `USE_CI_FILTER=False` |
| CI 门控（默认） | `USE_CI_FILTER=True`（见 §3A / `CI_FILTER_CHANGELOG.md`） |

固定 **seed / 数据集 / epoch 数 / GPU**；PAL 的主动学习池演化随机性大，
建议至少 2 个 seed 取均值，否则容易把波动误判为增益。

### 5.3 指标

- 训练/验证：`IoU`、`nIoU`、`Pd`、`Fa`（日志里 FA 打印时 ×1e6）
- 日志：`work_dirs/<MODEL_NAME>/<MODEL_NAME>.txt`
- 权重：`work_dirs/<MODEL_NAME>/best_mIoU_checkpoint_<MODEL_NAME>.pth.tar`

### 5.4 阶段划分（`NUM_EPOCHS=400` 时）

| 阶段 | epoch | 行为 |
|---|---|---|
| pre-start | 0–80 | 只用简单样本，不更新伪标签 |
| enhancement | 80–320 | 每 5 epoch 做 COU（挑难样本）+ FIU（精炼伪标签） |
| refinement | 320–400 | 只做 FIU |

前 `num_start_test_epochs=100` 个 epoch **不做验证**。快速验证可调小 `NUM_EPOCHS`，
但 baseline 与 TATE 必须同步调。

---

## 6. 已知坑 / 已修复

| # | 问题 | 状态 |
|---|---|---|
| 1 | MSDA 的 DWT 反向硬编码 `.to(dtype=torch.float16)`，fp32 反向必崩 `expected scalar type Float but found Half` | **已修** → `.to(dtype=dx.dtype)`（5 处）。AMP 下与原行为等价 |
| 2 | `test_model.py` 原版不接文本先验 → TATE 的 checkpoint 加载报 unexpected keys，且**特征增强缺失、结果不可信** | **已修** → 同样接入，`strict=False` + 提示 |
| 3 | `text_prior.py` 的 `alpha` 是 fp32，AMP 下会把该层输出抬成 fp32 | **已修** → `self.alpha.to(feat.dtype)` |
| 4 | `CUDA_VISIBLE_DEVICES='2'` 硬编码 | 需人工改 |
| 5 | 伪标签池污染（见 §5.1） | 需人工隔离 |
| 6 | `work_dirs/` 里的 `.pth.tar` 权重很大，**目前未被 .gitignore 覆盖** | ⚠️ 见 §7 |
| 7 | `.gitignore` 里的 `./PAL-main/dataset` 写法不规范（实测仍生效，但建议改为 `/PAL-main/dataset/`） | 建议修 |

---

## 7. Git 版本管理约定

### 7.1 必须忽略（大文件 / 生成物）

```gitignore
# 数据集与工作池（790MB+，训练中还会变）
/PAL-main/dataset/
# 模型权重（每个几百 MB）
/PAL-main/work_dirs/**/*.pth.tar
/PAL-main/work_dirs/**/*.pth
# 可视化产物
/PAL-main/imgs/*.png
# 缓存
__pycache__/
*.py[cod]
.vscode/
```

> ⚠️ 当前 `.gitignore` **没有**忽略 `work_dirs/**/*.pth.tar`。
> 训练后 `git add -A` 会把几百 MB 权重提交进仓库，务必先补上。
> 训练日志 `.txt` 可以保留（体积小、有分析价值）。

### 7.2 应该提交

- 所有 `.py` 源码、`AGENTS.md`、`TEXT_PRIOR_README.md`
- 三个 `.patch` 文件（作为与原版 PAL 的差异存档）
- `work_dirs/**/*.txt` 训练日志（可选）

### 7.3 提交纪律

- **改完文本先验代码 → 先跑 `tools/test_text_prior.py`**，通过再提交。
- 每次实验（baseline / TATE / 消融）建议打 **tag** 或建 **branch**，并在 commit message
  里记录**关键开关值**（`USE_TEXT_PRIOR` / `TEXT_PRIOR_RANDOM` / `NUM_EPOCHS` / seed）。
  否则事后无法复现某个 checkpoint 是用什么配置训出来的。

---

## 8. 给后续 Agent 的注意事项

1. **不要擅自跑训练/推理**：需要 GPU、CLIP 权重、数据，单次完整训练耗时很长。
   除非用户明确要求，否则只做代码阅读/修改/静态检查。
2. **先确认改的是哪个项目**：本目录是 `PAL+TATE`，与 `SeViL`（mmdet）、`UCOD-MKD`
   （MLLM+SAM）完全独立，不要跨项目复制框架代码。
3. **改文本先验前先读 §3.4 的三个设计决策**，尤其是残差门控 `alpha`——
   直接去掉门控会让训练初期变差，容易被误判为"文本先验无效"。
4. **判断增益时务必排除"多了一个模块"的影响**：
   必须做 `TEXT_PRIOR_RANDOM=True` 的随机文本对照。
5. **扩展其他模型**：只需在 `MODEL_TEXT_PRIOR_CFG` 补 `layer_name` 与 `feat_dim`，
   并在接入前用 `forward hook` 验证该层输出形状。其他 7 个模型返回结构不统一
   （`DNA` 返回 4 元组、`UIU` 返回 7 元组），需逐个确认。
6. **本文件若与代码不一致，以代码为准，并顺手更新本文件。**
