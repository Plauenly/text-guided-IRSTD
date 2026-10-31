import torch
import torchvision
from components.dataset_final_edge_copy_paste_final_2_img_path import SirstDataset
from torch.utils.data import DataLoader
import os

def make_dir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    patch_size,
    train_batch_size,
    test_batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,       # 加快数据传输到显存的速度，代价是占用内存的大小变大
    train_pointdir=None,   # 点标注目录（训练时生成高斯热图用）
    crop_fov=None,         # 固定视野比例（见 SirstDataset docstring）
    aux_mode='edge',       # 'heatmap' | 'edge'
    gf_sigma=1.5,
    gf_sigma_mode='fixed',
    gf_sigma_min=1.0,
    gf_sigma_max=4.0,
):
    train_ds = SirstDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        patch_size=patch_size,
        transform=train_transform,
        mode='train',
        point_dir=train_pointdir,
        crop_fov=crop_fov,
        aux_mode=aux_mode,
        sigma=gf_sigma,
        sigma_mode=gf_sigma_mode,
        sigma_min=gf_sigma_min,
        sigma_max=gf_sigma_max,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = SirstDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        patch_size=patch_size,
        transform=val_transform,
        mode='val',
    )

    # 验证集保持原图尺寸（只 pad 到 32 的倍数），不同图尺寸不同，因此 batch=1。
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

