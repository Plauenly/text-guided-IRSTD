# -*- coding: utf-8 -*-
"""
文本先验特征增强模块（Text-guided Adaptive Target Enhancing, TATE 风格）

设计来源：SeViL (AAAI'26) 的 TextGuidedAttention2
    SeViL/mmdet/models/necks/fpn.py:663-767

本文件针对 PAL（单帧 + 点监督）所做的改造：
    1. 通道自适应：SeViL 中文本嵌入与视觉特征同为 256 维，可直接 expand_as；
       PAL 的 MSDA bottleneck 只有 64 维，因此 mlp_vis 输出维度改为 feat_dim。
    2. 文本嵌入预先编码并缓存为 buffer：SeViL 每次 forward 都重新 encode_text，
       这里只在初始化时算一次，训练零额外开销。
    3. 增加可选的残差门控 alpha（默认开启，初值 0）：
       保证训练初期该模块等价于恒等映射，不会破坏 baseline 的收敛，
       再让网络自己学习"要不要用文本先验"。
    4. 零侵入接入：通过 forward hook 注入到指定层，不修改任何模型源码。

数值流程（与论文 Eq.1-3 对应）：
    E  = CLIP_text(prompts)                     # [N, 512]
    Ep = f_exp( mean_i( f_mlp(E_i) ) )          # f_mlp: 线性层, f_exp: 空间广播
    G  = Softmax( f_c( M ⊙ Ep ) )               # f_c: 1x1 卷积 + 空间 softmax
    M_E = f_c( G ⊙ M + M )                      # f_c: 1x1 卷积
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. CLIP 文本嵌入器：加载 CLIP 文本塔，把 prompt 列表编码成 E ∈ R^{N×D}
# =============================================================================
class ClipTextEmbedder:
    """加载 CLIP，把固定 prompt 编码为文本嵌入并缓存；编码后立即释放 CLIP 以省显存。

    Args:
        prompts (list[str]): 文本 prompt 列表，长度为 N。
        model_name_or_path (str): 'ViT-B/32' 之类的模型名（会自动下载），
            或本地权重文件路径（如 '/path/to/ViT-B-32.pt'）。
        device (str): 'cuda' 或 'cpu'。
        keep_image_tower (bool): 是否保留 CLIP 图像塔。
            True 时把整个 CLIP 模型保存在 `self.clip_model` 里不释放，供
            CI 门控（components/credibility.py）计算图文相似度复用同一份
            文本嵌入；关闭时行为与原来完全一致（编码完立即释放）。
        verbose (bool): 是否打印加载信息。
    """

    def __init__(self, prompts, model_name_or_path='ViT-B/32', device='cuda',
                 keep_image_tower=False, verbose=True):
        try:
            import clip
        except ImportError as e:
            raise ImportError(
                "\n[TextPrior] 未找到 `clip` 包。请先安装 openai-clip：\n"
                "    pip install ftfy regex\n"
                "    pip install git+https://github.com/openai/CLIP.git\n"
                "或参见 TEXT_PRIOR_README.md 的离线权重说明。\n"
            ) from e

        if len(prompts) == 0:
            raise ValueError('[TextPrior] prompts 不能为空。')

        self.device = device
        if verbose:
            print(f'[TextPrior] 加载 CLIP 文本塔: {model_name_or_path} (device={device})')

        # clip.load 在给模型名时自动下载到 ~/.cache/clip；给本地路径时直接加载
        clip_model, _ = clip.load(model_name_or_path, device=device)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad_(False)

        tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(tokens)   # [N, 512]，CLIP 默认 fp16

        # 统一转 fp32 保存
        self.text_features = text_features.detach().float().cpu()

        if keep_image_tower:
            # CI 门控需要图像塔：保留模型（已在 eval + requires_grad=False）
            self.clip_model = clip_model
            if verbose:
                print(f'[TextPrior] 文本嵌入已缓存: shape={tuple(self.text_features.shape)}，'
                      f'CLIP 图像塔保留（供 CI 门控复用）。')
        else:
            # 原行为：CLIP 之后不再需要，立即释放显存
            self.clip_model = None
            del clip_model
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            if verbose:
                print(f'[TextPrior] 文本嵌入已缓存: shape={tuple(self.text_features.shape)}，'
                      f'CLIP 已释放。')

    def get(self):
        """返回缓存的文本嵌入 [N, 512]（CPU, float32）。"""
        return self.text_features

    def get_image_tower(self):
        """返回保留的 CLIP 模型（含 encode_image）；未保留时返回 None。"""
        return getattr(self, 'clip_model', None)


# =============================================================================
# 2. 文本先验注意力模块（TATE）
# =============================================================================
class TextPriorAttention(nn.Module):
    """用文本先验生成空间注意力图，对视觉特征做残差加权增强。

    Args:
        feat_dim (int): 被增强特征的通道数（MSDA bottleneck = 64）。
        text_dim (int): 文本嵌入维度（CLIP ViT-B/32 = 512）。
        hidden_dim (int): f_mlp 隐层维度。
        use_gate (bool): 是否使用残差门控 alpha（推荐 True，训练更稳）。
        gate_init (float): alpha 初值，0 表示初始等价恒等映射。
    """

    def __init__(self, feat_dim, text_dim=512, hidden_dim=256,
                 use_gate=True, gate_init=0.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.text_dim = text_dim
        self.use_gate = use_gate

        # f_mlp: 文本 -> 视觉特征子空间（输出维度对齐 feat_dim，便于逐元素相乘）
        self.mlp_vis = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, feat_dim),
        )

        # f_c: 相似度图 -> 单通道注意力
        self.similarity_mlp = nn.Conv2d(feat_dim, 1, 1, bias=False)
        # f_c: 融合后的通道重标定
        self.reweight = nn.Conv2d(feat_dim, feat_dim, 1, bias=False)

        # 文本嵌入缓存；初始为空，随后由 set_text_embedding 填入
        self.register_buffer('text_embedding', torch.zeros(0, text_dim), persistent=False)

        if use_gate:
            self.alpha = nn.Parameter(torch.tensor(float(gate_init)))

        self._last_attention = None   # 便于调试/可视化

    # ------------------------------------------------------------------ #
    def set_text_embedding(self, text_embedding):
        """填入预编码的文本嵌入 E ∈ R^{N×D}。"""
        assert text_embedding.dim() == 2 and text_embedding.size(1) == self.text_dim, \
            f'期望文本嵌入形状 [N,{self.text_dim}]，实际得到 {tuple(text_embedding.shape)}'
        self.text_embedding = text_embedding.detach().float()

    # ------------------------------------------------------------------ #
    def compute_attention(self, feat):
        """计算注意力图 G（Eq.1 + Eq.2）。

        Args:
            feat: [B, feat_dim, H, W]
        Returns:
            attention_map: [B, 1, H, W]，空间维 softmax 归一化
        """
        # ---- Eq.1: E -> f_mlp -> 平均 -> 广播为 E_p ----
        text = self.text_embedding.to(device=feat.device, dtype=torch.float32)
        text = self.mlp_vis(text)                      # [N, feat_dim]
        text = text.mean(dim=0)                        # [feat_dim]  (1/N)Σ
        text = text.view(1, -1, 1, 1)
        text = text.expand_as(feat)                    # [B, feat_dim, H, W]

        # ---- Eq.2: M ⊙ E_p -> 1x1 conv -> 空间 softmax ----
        similarity = feat.to(torch.float32) * text     # [B, feat_dim, H, W]
        attention_map = self.similarity_mlp(similarity)  # [B, 1, H, W]
        attention_map = F.softmax(
            attention_map.flatten(2), dim=-1
        ).view_as(attention_map)                       # G
        return attention_map

    # ------------------------------------------------------------------ #
    def forward(self, feat):
        """增强视觉特征。

        Args:
            feat: [B, feat_dim, H, W]
        Returns:
            enhanced: [B, feat_dim, H, W]
        """
        attention_map = self.compute_attention(feat)
        self._last_attention = attention_map.detach()

        # ---- Eq.3: M_E = f_c(G ⊙ M + M) ----
        weighted = attention_map.to(feat.dtype) * feat + feat
        weighted = self.reweight(weighted)

        if self.use_gate:
            # 残差门控：alpha=0 时输出 == 输入（恒等），保证不破坏 baseline
            # 注意：把 alpha 转成 feat.dtype，避免 AMP 下 fp32 参数把该层输出抬成 fp32
            enhanced = feat + self.alpha.to(feat.dtype) * (weighted - feat)
        else:
            enhanced = weighted
        return enhanced


# =============================================================================
# 3. 零侵入接入：通过 forward hook 把文本注意力插到模型的指定层
# =============================================================================
# 每个模型需要指定：注入层的模块路径、该层输出通道数
MODEL_TEXT_PRIOR_CFG = {
    # MSDA: Stage 的最后一层 layer5_4 输出即 bottleneck（64 通道）
    # 调用链：MSDA_No_Sigmoid.stage(Stage).layer5_4
    'MSDA': dict(layer_name='stage.layer5_4', feat_dim=64),
    # 其他模型待扩展（需要先确认其 bottleneck 的模块路径与通道数）
    # 'ACM':  dict(layer_name='...', feat_dim=...),
    # 'ALC':  dict(layer_name='...', feat_dim=...),
    # 'MLCL': dict(layer_name='...', feat_dim=...),
    # 'ALCL': dict(layer_name='...', feat_dim=...),
    # 'GGL':  dict(layer_name='...', feat_dim=...),
    # 'DNA':  dict(layer_name='...', feat_dim=...),
    # 'UIU':  dict(layer_name='...', feat_dim=...),
}


def attach_text_prior(model, text_embedding, choose_model='MSDA',
                      hidden_dim=256, use_gate=True, gate_init=0.0,
                      verbose=True):
    """把 TextPriorAttention 以 forward hook 形式挂到模型上（不修改模型源码）。

    注意：模块通过 model.add_module() 注册，因此 model.parameters() 会自动包含它，
          train_model.py 里原有的 optimizer = AdamW(model.parameters()) 无需改动。

    Args:
        model (nn.Module): 已构建好的分割网络（如 MSDA_No_Sigmoid）。
        text_embedding (Tensor): ClipTextEmbedder.get() 的返回值，[N, 512]。
        choose_model (str): 模型名，用于查 MODEL_TEXT_PRIOR_CFG。
        hidden_dim (int): f_mlp 隐层维度。
        use_gate (bool): 是否使用残差门控。
        gate_init (float): 门控初值。
        verbose (bool): 是否打印接入信息。

    Returns:
        (TextPriorAttention, hook_handle)
    """
    if choose_model not in MODEL_TEXT_PRIOR_CFG:
        raise NotImplementedError(
            f'[TextPrior] 模型 {choose_model} 尚未配置注入层。'
            f'请在 components/text_prior.py 的 MODEL_TEXT_PRIOR_CFG 中补充 '
            f'layer_name 与 feat_dim。目前已支持: {list(MODEL_TEXT_PRIOR_CFG.keys())}'
        )

    cfg = MODEL_TEXT_PRIOR_CFG[choose_model]
    text_attn = TextPriorAttention(
        feat_dim=cfg['feat_dim'],
        text_dim=text_embedding.size(1),
        hidden_dim=hidden_dim,
        use_gate=use_gate,
        gate_init=gate_init,
    )
    text_attn.set_text_embedding(text_embedding)

    # 与主模型放到同一 device（model.to(DEVICE) 之后再 add_module 不会自动迁移）
    try:
        model_device = next(model.parameters()).device
        text_attn = text_attn.to(model_device)
    except StopIteration:
        pass

    # 注册为子模块，使 model.parameters() 覆盖其参数
    model.add_module('text_prior', text_attn)

    # 定位目标层
    target = model
    for attr in cfg['layer_name'].split('.'):
        target = getattr(target, attr)

    # forward hook：用模块输出替换该层的输出
    def _hook(layer, inputs, output):
        return text_attn(output)

    handle = target.register_forward_hook(_hook)

    if verbose:
        n_params = sum(p.numel() for p in text_attn.parameters())
        print(f'[TextPrior] 已接入 {choose_model}: '
              f'注入层={cfg["layer_name"]}, feat_dim={cfg["feat_dim"]}, '
              f'可训练参数={n_params} (~{n_params/1e3:.2f}K), '
              f'gate={use_gate}(init={gate_init})')
    return text_attn, handle
