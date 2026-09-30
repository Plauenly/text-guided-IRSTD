# -*- coding: utf-8 -*-
"""
单帧红外小目标 —— 文本 prompt 定义（文本先验知识）

来源：SeViL (AAAI'26) 的 prompt 列表（见 SeViL/mmdet/models/necks/fpn.py:678-724 的 Target3 版本）。
本文件仅做「单帧化裁剪」：删除所有依赖多帧/运动/时序的条目，保留下来的表述本身不变。

被删除的 SeViL 原始 prompt（多帧专属，单帧设定下不成立）：
    1) "The target may show slight movement, often revealing a simple trajectory in
        frames, but may be jittery due to background interference."
       —— 依赖帧间运动。
    2) "The target's brightness intensity may slightly fluctuate across frames but
        overall the target region will still be brighter than the background."
       —— 依赖跨帧亮度波动。
    3) "The target may maintain a certain stability in multi-frame images, but slight
        jitter may occur due to noise and background interference."
       —— 依赖多帧稳定性。

其余 prompt 原样保留（顺序与 SeViL 保持一致，便于对照）。
"""

# =============================================================================
# 主用 prompt 集（10 条，单帧适用）
# =============================================================================
SINGLE_FRAME_IR_PROMPTS = [
    # 亮度 / 温度：目标通常比背景亮，温差明显
    "The target is usually brighter than the surrounding background, with a higher temperature, making it stand out in the image.",
    # 形状：受噪声干扰形状不规则、边缘模糊
    "The target shape is irregular due to background noise and surrounding interference, showing blurred edge contours.",
    # 尺寸：只占几个到十几个像素
    "The target occupies only a few pixels in the image, typically ranging from several to a dozen pixels.",
    # 对比度：与背景有对比度，因而更突出
    "The target often has a contrast with the background, making it more prominent in the image.",
    # 热源特性：与背景温度差异显著
    "The target typically as a thermal source has distinct temperature characteristics, with a significant difference from the background temperature.",
    # 孤立性：周围没有其他显著特征点
    "The target is usually isolated in the image, with no other significant feature points around it.",
    # 易被淹没：亮度和信号弱于背景噪声
    "The target is prone to being obscured, due to the low brightness and the weak signal compared to background noise.",
    # 纹理相似：灰度模糊，与背景纹理相近
    "The target has a surface texture similar to the background, appearing gray and blurry, indicating its temperature.",
    # 语义类别：热源/飞机/车辆等感兴趣目标
    "The target is specific types of objects, such as heat sources, aircraft, vehicles, etc., that are parts of interest in the scene.",
    # 信噪比：信噪比低，难以从背景噪声中检测
    "The target typically has a low signal-to-noise ratio, making it harder to detect from background noise.",
]


# =============================================================================
# 备用：精简 prompt 集（4 条）—— 用于 prompt 数量的消融实验
# =============================================================================
SINGLE_FRAME_IR_PROMPTS_MINIMAL = [
    "The target is usually brighter than the surrounding background, with a higher temperature, making it stand out in the image.",
    "The target occupies only a few pixels in the image, typically ranging from several to a dozen pixels.",
    "The target typically has a low signal-to-noise ratio, making it harder to detect from background noise.",
    "The target is usually isolated in the image, with no other significant feature points around it.",
]


# =============================================================================
# 备用：只保留"尺寸 + 亮度"两条（最保守的先验）
# =============================================================================
SINGLE_FRAME_IR_PROMPTS_TINY = [
    "The target is usually brighter than the surrounding background, with a higher temperature, making it stand out in the image.",
    "The target occupies only a few pixels in the image, typically ranging from several to a dozen pixels.",
]
