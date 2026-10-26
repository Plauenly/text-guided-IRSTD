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
    def __init__(self, image_dir, mask_dir, patch_size, transform=None, mode='None'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = np.sort(os.listdir(image_dir))
        self.mode = mode
        self.patch_size = patch_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)  
        mask = (mask > 127.5).astype(float)

        # 新流程的伪标签始终与原图同尺寸；这里只兼容旧版 256×256 伪标签。
        mask = match_mask_to_image(mask, image)

        if (self.mode == 'train'):
            # 恢复原始 PAL 的随机裁剪：输出固定 patch_size(256×256) 的 patch，batch_size>1 才能正常拼接。
            # 注意：train_transform 里并没有随机裁剪，所以这里必须保留 random_crop。
            image_patch, mask_patch = random_crop(image, mask, self.patch_size)
            if self.transform is not None:
                augmentations = self.transform(image=image_patch, mask=mask_patch)
                image = augmentations["image"]
                mask = augmentations["mask"]
            mask_2 = mask.numpy()       # transform变换中使用了totensor方法，所以得使用.numpy()
            mask_2 = mask_2.astype(np.int64)
            oneHot_label = mask_to_onehot(mask_2, 2)    # 该函数获得前景与背景中的掩码，即oneHot_label.shape[0] = 2
            edge = onehot_to_binary_edges(oneHot_label, 1, 2)
            edge[1, :] = 0
            edge[-1:, :] = 0
            edge[:, :1] = 0
            edge[:, -1:] = 0
            edge = np.expand_dims(edge, axis=0).astype(np.int64)
            return image, mask, edge

        elif (self.mode == 'val'):
            # 只 pad 到 32 的倍数，不 resize；返回原图 h,w，供评估时裁回原坐标。
            image, mask, h, w = pad_to_multiple(image, mask, multiple=32)
            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]

            return image, mask, h, w
