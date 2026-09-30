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
└── PAL-main/                  ← 实际代码
    ├── train_model.py         ★ 训练主循环 + 主动学习调度 + 文本先验接入
    ├── test_model.py          ★ 测试（已支持文本先验）
    ├── utilts.py              ★ EPG 初始伪标签生成 / 难易样本划分 / 伪标签精炼
    ├── components/
    │   ├── text_prior.py      ★★ 文本先验模块（TATE）—— 本分支核心新增
    │   ├── text_prompts.py    ★★ 单帧红外 prompt 列表
    │   ├── dataset_final_edge_copy_paste_final_2_img_path.py  数据集（返回 img/mask/edge）
    │   ├── utils_all_edge_copy_paste_final_2_img_path.py      DataLoader 封装
    │   ├── metric_new_crop.py 指标（IoU / nIoU / Pd / Fa）
    │   ├── edges.py, gardient.py, cal_mean_std.py
    ├── loss/Edge_loss.py      ★ EEDM 损失（edgeSCE_loss）
    ├── model/{MSDA,ACM,ALC,MLCL,ALCL,DNA,GGL,UIU}/*_no_sigmoid.py   8 个可选网络
    ├── tools/
    │   ├── test_text_prior.py ★ 文本先验自检脚本（不依赖数据集）
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
| `tools/test_text_prior.py` | **新增** | 自检脚本（模块 / MSDA 接入 / CLIP 编码） |
| `PAL-main/TEXT_PRIOR_README.md` | **新增** | 使用说明 |
| `train_model.py` | 改 | 顶部 7 个开关；模型构造后接入文本先验 |
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
