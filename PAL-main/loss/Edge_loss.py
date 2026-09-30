#!/usr/bin/python3
# coding = gbk
"""
@Author : yuchuang
@Time : 2024/3/30 22:07
@desc:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss, LovaszLoss,SoftBCEWithLogitsLoss

def edgeSCE_loss(pred, target, edge):

    BinaryCrossEntropy_fn = SoftBCEWithLogitsLoss(smooth_factor=None, reduction='None') #当参数smooth_factor=None时，该类等价于BCEWithLogitsLoss类
    # 该损失函数，将每个像素点是为二分类任务，对每个像素点使用二分类交叉熵，注意pred必须是没有经过激活层的，因为该函数内部自带激活。
    edge_weight = 4.
    loss_sce = BinaryCrossEntropy_fn(pred, target)
    #print(loss_sce.size())
    #print(edge.size())
    edge = edge.clone()             # edge是边缘掩码，=0表示非边缘，>0表示边缘
    edge[edge == 0] = 1.
    edge[edge > 0] = edge_weight    # 给予边缘位置更高的损失权重
    loss_sce *= edge  # loss_sce.size =b,c,h,w -> b*c*h*w

    loss_sce_, ind = loss_sce.contiguous().view(-1).sort()  # .view(-1)函数将张量延展为1维
    min_value = loss_sce_[int(0.5 * loss_sce.numel())]      # .numel()函数表示元素个数 , min_value = 所有元素的中位数
    loss_sce = loss_sce[loss_sce >= min_value]              # loss_sce >= min_value得到二值掩码T，loss_sce[T]根据布尔索引筛选出相应的元素    
    loss_sce = loss_sce.mean()
    loss = loss_sce

    return loss

# if __name__ == '__main__':
#     target=torch.ones((2,1,256,256),dtype=torch.float32)
#     input=(torch.ones((2,1,256,256))*0.9)
#     input[0,0,0,0] = 0.99
#     loss=edgeBCE_Dice_loss(input,target,target*255)
#     print(loss)

