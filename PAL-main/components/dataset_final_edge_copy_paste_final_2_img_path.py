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

        # 关键修复：自更新写回的 mask 被 resize 成 256×256，而原图尺寸可能不同。
        # 统一把 mask 对齐到原图尺寸，避免 random_crop / albumentations 的形状断言失败。
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

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
            # times = 32
            # h, w, c = image.shape
            # pad_height = math.ceil(h / times) * times - h
            # pad_width = math.ceil(w / times) * times - w
            # image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
            # mask = np.pad(mask, ((0, pad_height), (0, pad_width)), mode='constant')
            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]

            return image, mask
