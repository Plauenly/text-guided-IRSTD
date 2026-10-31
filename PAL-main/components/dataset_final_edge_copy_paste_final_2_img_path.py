import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import time
from components.edges import onehot_to_binary_edges, mask_to_onehot
from components.heatmap import gaussian_heatmap, point_centers
import torch
import cv2
import random
import math


_LEGACY_MASK_WARNED = False


def pad_to_multiple(image, mask=None, multiple=32):
    """只做右/下 padding，不做 resize；返回 padding 后的 image/mask 与原图 h,w。"""
    h, w = image.shape[:2]
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        if mask is not None:
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
    return image, mask, h, w


def match_mask_to_image(mask, image):
    """新伪标签应始终与原图同尺寸；这里只兼容旧版 256×256 伪标签。"""
    global _LEGACY_MASK_WARNED
    if mask.shape[:2] == image.shape[:2]:
        return mask
    if mask.shape[:2] == (256, 256):
        if not _LEGACY_MASK_WARNED:
            print("[warn] 检测到旧版 256×256 伪标签，正在临时对齐到原图；"
                  "建议删除旧伪标签目录后重新生成，才能获得原图坐标的标签。")
            _LEGACY_MASK_WARNED = True
        return cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    raise ValueError(f"mask 与原图尺寸不一致且不是旧版 256×256：image={image.shape[:2]}, mask={mask.shape[:2]}")



def random_crop(img, mask, patch_size):
    h, w, c = img.shape
    mh, mw = mask.shape

    assert (h, w) == (mh, mw), "Image and mask must have the same height and width"

    if min(h, w) < patch_size:
        img = np.pad(img, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w), (0, 0)), mode='constant')
        mask = np.pad(mask, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)), mode='constant')
        h, w, _ = img.shape

    h_start = random.randint(0, h - patch_size)
    h_end = h_start + patch_size
    w_start = random.randint(0, w - patch_size)
    w_end = w_start + patch_size

    img_patch = img[h_start:h_end, w_start:w_end, :]
    mask_patch = mask[h_start:h_end, w_start:w_end]

    return img_patch, mask_patch


class SirstDataset(Dataset):
    """训练/验证数据集。

    训练分支新增（2026-10 高斯 focal loss）：
      * `point_dir`：点标注目录。训练时与伪标签同时增强，用于生成高斯热图。
      * `crop_fov`：固定视野比例。>0 时先把整图缩放到 "crop 占原图 crop_fov"，
        再随机裁 `patch_size`；小图（不足 patch_size/crop_fov）则整图缩放到 patch_size。
        例：PATCH=320, crop_fov=0.8 → 512 图缩放 0.781（下采样），256 图缩放 1.25。
      * `aux_mode`：返回的第 3 个元素。'heatmap' = 高斯热图（新损失用）；
        'edge' = 原 PAL 的边缘图（回退用）。两者都是 3 元组，train_fn 不用改。
      * `sigma` / `sigma_mode`：热图生成参数（见 components/heatmap.py）。

    几何增强同时作用于"伪标签 + 点"：两者堆成 2 通道 mask 一起送进 albumentations，
    避免依赖 albumentations 的 `masks=[...]` 关键字（版本兼容性更稳）。
    点用最近邻变换，不会丢；放大/旋转后仍是 1 px。
    """

    def __init__(self, image_dir, mask_dir, patch_size, transform=None, mode='None',
                 point_dir=None, crop_fov=None, aux_mode='edge',
                 sigma=1.5, sigma_mode='fixed', sigma_min=1.0, sigma_max=4.0):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.point_dir = point_dir
        self.transform = transform
        self.images = np.sort(os.listdir(image_dir))
        self.mode = mode
        self.patch_size = patch_size
        self.crop_fov = crop_fov if (crop_fov is not None and crop_fov > 0) else None
        self.aux_mode = aux_mode
        self.sigma = sigma
        self.sigma_mode = sigma_mode
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __len__(self):
        return len(self.images)

    def _load_points(self, name, shape):
        """读点标注；没有 point_dir 时返回全 0（热图就是全 0）。"""
        if self.point_dir is None:
            return np.zeros(shape[:2], np.uint8)
        p_path = os.path.join(self.point_dir, name)
        pts = cv2.imread(p_path, cv2.IMREAD_GRAYSCALE)
        if pts is None:
            return np.zeros(shape[:2], np.uint8)
        if pts.shape[:2] != shape[:2]:
            pts = cv2.resize(pts, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return (pts > 127).astype(np.uint8) * 255

    def _resize_for_fov(self, image, mask, pts):
        """固定视野比例缩放（见类 docstring）。返回缩放后的 image/mask/pts。"""
        if self.crop_fov is None:
            return image, mask, pts
        H, W = image.shape[:2]
        side = min(H, W)
        if side >= self.patch_size / self.crop_fov:
            s = self.patch_size / (self.crop_fov * side)      # 大图：下采样
        else:
            s = self.patch_size / side                        # 小图：整图放大
        if abs(s - 1.0) < 1e-6:
            return image, mask, pts
        new_w, new_h = int(round(W * s)), int(round(H * s))
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        image = cv2.resize(image, (new_w, new_h), interpolation=interp)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # 点的坐标用数学缩放重建（不要把 1px 点图拿来重采样，会丢点）
        if pts is not None and (pts > 127).any():
            new_pts = np.zeros((new_h, new_w), np.uint8)
            for (cx, cy) in point_centers(pts):
                nx, ny = int(round(cx * s)), int(round(cy * s))
                if 0 <= nx < new_w and 0 <= ny < new_h:
                    new_pts[ny, nx] = 255
            pts = new_pts
        else:
            pts = np.zeros((new_h, new_w), np.uint8)
        return image, mask, pts

    def _crop_patch(self, image, mask, pts):
        """右/下 pad 到至少 patch_size，再随机裁 patch_size×patch_size。"""
        ps = self.patch_size
        H, W = image.shape[:2]
        pad_h, pad_w = max(0, ps - H), max(0, ps - W)
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')
            pts = np.pad(pts, ((0, pad_h), (0, pad_w)), mode='constant')
            H, W = image.shape[:2]
        y0 = random.randint(0, H - ps)
        x0 = random.randint(0, W - ps)
        return (image[y0:y0 + ps, x0:x0 + ps],
                mask[y0:y0 + ps, x0:x0 + ps],
                pts[y0:y0 + ps, x0:x0 + ps])

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)  
        mask = (mask > 127.5).astype(float)

        # 新流程的伪标签始终与原图同尺寸；这里只兼容旧版 256×256 伪标签。
        mask = match_mask_to_image(mask, image)

        if (self.mode == 'train'):
            pts = self._load_points(self.images[index], image.shape)
            # 1) 固定视野比例缩放；2) pad + 随机裁剪
            image, mask, pts = self._resize_for_fov(image, mask, pts)
            image_patch, mask_patch, pts_patch = self._crop_patch(image, mask, pts)

            # 3) 几何增强：伪标签 + 点堆成 2 通道 mask 一起变换（保证逐像素对齐）
            mask2 = np.stack([(mask_patch > 0.5).astype(np.uint8) * 255,
                              pts_patch], axis=-1)          # (H,W,2)
            if self.transform is not None:
                aug = self.transform(image=image_patch, mask=mask2)
                image = aug["image"]
                mask_out = aug["mask"]
            else:
                mask_out = mask2
            if hasattr(mask_out, 'numpy'):          # ToTensorV2 -> tensor
                mask_out = mask_out.numpy()
            if mask_out.ndim == 2:                  # 兜底：万一通道被压掉
                mask_out = mask_out[..., None]
            pseudo = (mask_out[..., 0] > 127).astype(np.float32)     # (H,W) 0/1
            # 注意 mask_out 是 uint8（0/255），不能再乘 255（会溢出）
            points_aug = (mask_out[..., 1] if mask_out.shape[-1] > 1
                          else np.zeros(pseudo.shape, np.uint8)).astype(np.uint8)

            # train_fn 里会对 target 做 unsqueeze(1)，所以这里返回 (H,W)
            target = torch.from_numpy(pseudo)

            if self.aux_mode == 'heatmap':
                hm = gaussian_heatmap(points_aug, sigma=self.sigma,
                                      sigma_mode=self.sigma_mode,
                                      label_mask=pseudo,
                                      sigma_min=self.sigma_min,
                                      sigma_max=self.sigma_max)
                aux = torch.from_numpy(hm).unsqueeze(0)        # (1,H,W) float32
                return image, target, aux

            # ---- 回退：原 PAL 的边缘图 ----
            mask_2 = pseudo.astype(np.int64)
            oneHot_label = mask_to_onehot(mask_2, 2)
            edge = onehot_to_binary_edges(oneHot_label, 1, 2)
            edge[:1, :] = 0        # 原代码这里写成 edge[1,:]，会漏掉第 0 行，已修正
            edge[-1:, :] = 0
            edge[:, :1] = 0
            edge[:, -1:] = 0
            edge = np.expand_dims(edge, axis=0).astype(np.int64)
            return image, target, edge

        elif (self.mode == 'val'):
            # 只 pad 到 32 的倍数，不 resize；返回原图 h,w，供评估时裁回原坐标。
            image, mask, h, w = pad_to_multiple(image, mask, multiple=32)
            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]

            return image, mask, h, w
