# PAL + TATE：文本先验特征增强（快速验证版）

本目录 = 原版 PAL 的一份副本，**唯一的新增**是：把 SeViL 的文本先验（TATE）以
**零侵入 forward hook** 的方式注入到 **MSDA** 的 bottleneck，用来观察文本知识对模型的增益。

原版 PAL 逻辑完全保留（`USE_TEXT_PRIOR = False` 即等价于 baseline）。

---

## 1. 本版本改了什么

| 文件 | 状态 | 说明 |
|---|---|---|
| `components/text_prompts.py` | **新增** | 单帧红外 prompt 列表（由 SeViL 的 13 条裁剪为 10 条，删除多帧/运动相关条目） |
| `components/text_prior.py` | **新增** | `ClipTextEmbedder` + `TextPriorAttention`(TATE) + `attach_text_prior()` |
| `train_model.py` | **改动** | 顶部加 7 个开关；模型构造后接入文本先验 |
| `test_model.py` | **改动** | 同样接入文本先验（否则测试时特征增强缺失，结果不可比）；`load_state_dict` 改为 `strict=False` 并给出不匹配提示 |
| 其他所有文件 | 未动 | 与原版一致 |

> ⚠️ **测试必须与训练一致**：TATE 训练的 checkpoint 含 `text_prior.*` 参数，
> 且前向依赖文本增强。`test_model.py` 已相应修改，并新增 `USE_TEXT_PRIOR` 开关。
> 若要测 baseline（未接文本先验）的 checkpoint，把 `USE_TEXT_PRIOR = False`。

> ⚠️ **伪标签池必须隔离**：池目录（`dataset/<数据集>/<MODEL_NAME>/`）在训练中会被
> **原地修改**（choose/mask 被覆盖、no_choose 被搬空）。跑 baseline 与 TATE 对照时，
> 两者必须从**同一份初始池**出发（删掉重生成，或改 `MODEL_NAME` 后缀隔离），
> 否则结论无效。

### 注入位置

```
MSDA_No_Sigmoid
 └── stage (Stage)
      └── layer5_4  ←── forward hook 注入 TextPriorAttention（64 通道 bottleneck）
```

### 关键设计

1. **文本侧可训练**：`f_mlp` / `similarity_mlp` / `reweight` 都可训练；CLIP 冻结且只用于
   把 prompt 编码一次（初始化时缓存为 buffer，训练零额外开销）。
2. **残差门控** `alpha`（初值 0）：训练初期模块等价恒等映射，不会破坏 baseline 收敛，
   由网络自己决定要不要用文本先验。
3. **随机文本消融开关**：`TEXT_PRIOR_RANDOM=True` 用同形状随机向量，参数量完全一致，
   用来区分「文本真的携带信息」还是「只是多了一个模块」。

---

## 2. 安装依赖

本模块用 **openai-clip**（和 SeViL 一致）：

```bash
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git
```

> 如果不想用 git 安装，也可以直接把 SeViL 自带的 CLIP 代码加入 `PYTHONPATH`：
> ```bash
> pip install ftfy regex
> pip install -e /workspace/SeViL/CLIP
> ```

---

## 3. 下载 CLIP 权重（ViT-B/32）

### 方式 A：自动下载（推荐）

`clip.load('ViT-B/32')` 会自动下载权重并缓存到 `~/.cache/clip/ViT-B-32.pt`。
装好 `clip` 包后直接跑训练即可，首次运行联网下载一次（约 350MB）。

> **先检查是否已经下过**（如果你在宿主机跑过 SeViL，很可能已经有了）：
> ```bash
> ls -lh ~/.cache/clip/
> ```
> 若已有 `ViT-B-32.pt`，则完全不需要再下载。

### 方式 B：手动下载 .pt 权重（离线/网络不稳时）

```bash
mkdir -p ~/.cache/clip
wget -O ~/.cache/clip/ViT-B-32.pt \
  https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```

下载后有两种用法（二选一）：

- 放到默认缓存目录 `~/.cache/clip/ViT-B-32.pt`，然后照常用 `clip.load('ViT-B/32')`；
- 或把 `train_model.py` 里的 `TEXT_CLIP_MODEL` 改成绝对路径：
  ```python
  TEXT_CLIP_MODEL = '/your/path/to/ViT-B-32.pt'
  ```

> 注意：即使加载本地 `.pt` 文件，仍然需要安装 `clip` 这个 Python 包来提供模型定义代码。

### 校验是否可用

```bash
python tools/test_text_prior.py --with_clip
```

如果权重不在默认缓存目录，直接指定绝对路径：

```bash
python tools/test_text_prior.py --with_clip --clip_path /your/path/ViT-B-32.pt
```

---

## 4.5 已修复的原版 MSDA bug（与文本先验无关）

`model/MSDA/MSDA_no_sigmoid.py` 的 4 个 DWT 自定义反向函数里，原来有 5 处：

```python
filters = filters.repeat(C, 1, 1, 1).to(dtype=torch.float16)   # 硬编码 fp16
```

这导致**只要不是 AMP autocast 下的 fp16 反向**（例如 fp32 训练、或本自检脚本的
`loss.backward()`），就会报：

```
RuntimeError: expected scalar type Float but found Half
```

已改为跟随梯度 dtype，二者在 AMP 下完全等价，同时 fp32 也能跑：

```python
filters = filters.repeat(C, 1, 1, 1).to(dtype=dx.dtype)
```

> 注意：这是**原 PAL 就存在的潜在 bug**，只是原版一直用 `torch.cuda.amp.autocast()`
> 训练所以没暴露。本修改对 AMP 训练是行为等价的，不影响 baseline 对比。

---

## 4. 运行

```bash
cd /workspace/PAL+TATE/PAL-main
python train_model.py
```

需要保证 `/workspace/SeViL/` 的阅读笔记里提到的那些原版 PAL 注意事项（工作目录、数据集、
`epoch=100` 硬编码等）一致；本副本没有修这些原版问题。

---

## 5. 实验矩阵（建议）

| 实验 | 配置 | 目的 |
|---|---|---|
| baseline | `USE_TEXT_PRIOR = False` | 原版 PAL |
| +TATE | `USE_TEXT_PRIOR = True`, `TEXT_PRIOR_RANDOM = False` | 真实文本先验 |
| +TATE(random text) | `USE_TEXT_PRIOR = True`, `TEXT_PRIOR_RANDOM = True` | 消融：文本是否真有信息 |
| +TATE(no gate) | `TEXT_PRIOR_USE_GATE = False` | 消融：门控的作用 |

对比指标用 PAL 原有的 **IoU / nIoU / Pd / Fa**（`val_fn` 输出）。

> ⚠️ PAL 的主动学习池演化本身有随机性，**务必固定 seed、数据集、epoch 数**再比较，
> 且建议至少跑 2 个 seed 取均值，否则容易把波动误判为增益。

---

## 6. 调试

- 打印文本注意力图（看模型学到了哪些区域）：
  ```python
  attn = model.text_prior._last_attention   # [B,1,H,W]，每次 forward 后更新
  ```
- 打印可训练参数是否包含文本模块：
  ```python
  print(sum(p.numel() for p in model.text_prior.parameters()))
  ```
- 如果提示 `未找到 clip 包` / `模型 XX 尚未配置注入层`，按报错信息处理即可。
