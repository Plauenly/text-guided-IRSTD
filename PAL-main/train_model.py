from tqdm import tqdm
import torch.optim as optim
# from model.MSDA.MSDA_no_sigmoid import MSDANet_No_Sigmoid
from datetime import datetime
import torch
import os
import sys
import json
import numpy as np

# 第三方图像增强库albumentations，适用于cv2的使用场景(cpu上)，它有一个好处是image与bbox-mask, point-mask进行的缩放操作是同等（不需要考虑坐标转换）
import albumentations as A
from albumentations.pytorch import ToTensorV2


# 项目自定义的模块
from components.utils_all_edge_copy_paste_final_2_img_path import get_loaders, make_dir
from components.dataset_final_edge_copy_paste_final_2_img_path import pad_to_multiple, match_mask_to_image
from components.cal_mean_std import Calculate_mean_std
from components.metric_new_crop import *
from loss.Edge_loss import edgeSCE_loss
from loss.gaussian_focal_loss import GaussianFocalLoss
from utilts import *
from components.text_prior import ClipTextEmbedder, attach_text_prior
from components.text_prompts import SINGLE_FRAME_IR_PROMPTS
from components.credibility import (CIPatchScorer, extract_components, sample_mean_ci,
                                    spurious_ratio, gmm_threshold, fallback_threshold,
                                    admit_ratio)
from components.cross_modal import TextAdapter, attach_cross_modal

import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torch.autograd import Variable
import math
import argparse

choose_model = 'MSDA'   ##choose model in [ACM, ALC, MLCL, ALCL, DNA, GGL, UIU, MSDA]
# 根据choose_model的值去import，实现动态的导入模块，函数返回一个类对象，需要实例化才能使用
model_func = access_model(choose_model)


choose_dataset = 'SIRST3'  ## choose dataset in [SIRST3, IRSTD_1K_point, NUDT_SIRST_1_1_point, SIRST_1_1_point_new]
choose_dataset_type = 'masks_coarse' ## choose dataset_type in [mask, masks_coarse, masks_centroid]   "mask": "Full supervision"; "masks_centroid": "Centroid point supervision"; "mask_coarse": "Coarse point supervision"

## Hyperparameters etc.
LEARNING_RATE = 1e-3  #default: 1e-3;    When the choose_model = 'GGL' or 'ALCL', "LEARNING_RATE = 5e-4" can be considered to obtain relatively more stable training.  
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# !!!!! #
TRAIN_BATCH_SIZE = 16   #default: 16
TEST_BATCH_SIZE = 16
# !!!!! #

# 关注一下这个参数是在干什么？ 通常在测试时候用于大图像分块处理策略
TEST_PATCH_BATCH_SIZE = 16

NUM_EPOCHS = 400     #default: 400
num_start_test_epochs = 100   #default: 100
NUM_WORKERS = 4

IMAGE_SIZE = 256
PATCH_SIZE = 320        # 训练 patch（原 256）：320=16*20，能被 bottleneck 的 /16 整除
CROP_FOV = 0.8          # 固定视野：crop 占原图 80%（512图缩放 0.781，256图缩放1.25）
PATCH_SIZE_test = 1024
CP_PROBABILITY = 0.5
# lose_point_ratio = 0.2
lose_point_ratio_init = 0.2
alarm_point_ration = 5
clear_epoch_gap = 5
clear_inital_ratio = 0.2   #  default:0.2.   [0, clear_inital_ratio]: model pre-start phase
final_epoch_ratio = 0.8    # default:0.8.     [clear_inital_ratio, final_epoch_ratio] : model enhancement phase; [final_epoch_ratio,1]: model refinement phase
PIN_MEMORY = True
LOAD_MODEL = False
fixed_length = 120
thresh_Tb = 0.5
thresh_k = 0.5
degen = 0.97

## ==================== 文本先验（TATE）相关开关 ====================
## 只做特征增强的快速验证：文本先验经 CLIP 文本塔编码后，通过 forward hook
## 注入到 MSDA 的 bottleneck(stage.layer5_4)，对特征做注意力式残差增强。
USE_TEXT_PRIOR = True          # False -> 完全等价于原版 PAL（baseline）
TEXT_PRIOR_RANDOM = False      # True  -> 用随机文本嵌入（关键消融：验证文本是否真的携带信息）
TEXT_CLIP_MODEL = 'ViT-B/32'   # 或本地权重路径，如 '/path/to/ViT-B-32.pt'
TEXT_PRIOR_HIDDEN = 256        # f_mlp 隐层维度
TEXT_PRIOR_USE_GATE = True     # 残差门控 alpha，保证初期不破坏 baseline
TEXT_PRIOR_GATE_INIT = 0.0     # alpha 初值，0 = 初始等价恒等映射
## =================================================================

## ============ CI 门控（TAPF 风格困难样本筛选）相关参数 ============
## 用 SeViL 的 CI = 2SK/(S+K) + GMM 动态阈值，替换 PAL 原有的
## "点位漏检率 / 虚警比" 人工判据（utilts.nc_pred_mask + deal_pred_mask_and_true_point_in）。
## 规则：对每张困难样本算 图级均值 CI（未匹配点记 0）→ 对全体困难样本的均值
##       拟合 2 分量 GMM 取高簇密度峰值作为 theta → mean_ci >= theta 才进简单池。
## 默认启用；置 False 可完全回到原版 PAL 门控（仅用于 A/B 对照）。
## 完整改动记录见 PAL+TATE/CI_FILTER_CHANGELOG.md
USE_CI_FILTER = True
CI_EMA_BETA = 0.5          # 阈值跨轮 EMA 平滑系数（越大越平滑，防抖动）
CI_MIN_POOL = 4            # 参与 GMM 的样本数少于此值时改用回退阈值
CI_FALLBACK = 'median'     # 回退阈值口径：'median'（池中位数）或 'fixed'（0.3）
## 数量闸：theta 只管"谁比谁好"，k 管"这轮最多放几个"（单峰分布时也能节流）
CI_USE_QUOTA = True        # False -> 只靠 theta（此时容易恢复"一次进太多"）
CI_ADMIT_RATIO_INIT = 0.05 # 每轮最多放行比例（占困难池）的起点，线性升到 1.0
CI_CROP_SIDE_RATIO = 1.5   # CLIP 局部窗边长 = 系数 × 过点组件 bbox 长边
CI_CROP_MIN = 16           # 局部窗边长下限（像素）：红外目标只有几个像素
CI_CROP_MAX = 64           # 局部窗边长上限（像素）
## =================================================================

## ============ 高斯 Focal Loss（替换 edgeSCE）相关参数 ============
## 设计：target 仍是**伪标签**（与 PAL 训练语义一致），高斯热图只做权重：
##   target=1: (1+(W-1)g)·(1-p)^α·softplus(-z)
##   target=0: (1-g)^γ·p^α·softplus(z)
## 归一化 by_pos_neg（分母 ≈ 有效负像素数）→ 损失量级与 patch 尺寸无关，
## 与旧 edgeSCE 同档（≈0.1~1），ACM 的 λ 不需要重标。详见 loss/gaussian_focal_loss.py
LOSS_TYPE = 'gauss_focal'      # 'gauss_focal' | 'edgeSCE'（回退到原 PAL）
GF_SIGMA_MODE = 'fixed'        # 'fixed' | 'from_label'
GF_SIGMA_PX = 1.5              # 高斯核宽度（patch 像素）
GF_SIGMA_MIN = 1.0
GF_SIGMA_MAX = 4.0
GF_ALPHA = 2.0                 # focal 正负调制指数 (1-p)^α / p^α
GF_GAMMA = 4.0                 # 负样本高斯衰减指数 (1-g)^γ
GF_POS_WEIGHT = 4.0            # W：核内正向最大权重（对应原 PAL edge_weight=4）
GF_NORM = 'by_pos_neg'         # 'by_pos_neg' | 'by_area' | 'by_pos'
GF_TARGET = 'pseudo'           # 'pseudo' | 'heatmap'（后者=CornerNet 原式，仅对照）
GF_POS_EPS = 1e-4
GF_SKIP_EMPTY = True           # 不含目标的 crop 不参与损失
GF_GRAD_CLIP = 1.0             # 梯度裁剪（None/0 关闭）
## =================================================================

## ============ 文本适配器 + 跨模态掩码（ACM）相关参数 ============
## 设计: Ea = E + gate*MLP(E)（gate 零初始化）；ACM 把 MSDA 融合前的 128ch/16x16
## 特征投影到 512 维，用注意力从视觉特征里重建被掩码的文本语义。
## 公式与数值细节见 components/cross_modal.py 文件头；改动记录见 CI_FILTER_CHANGELOG.md
USE_TEXT_ADAPTER = True        # 文本适配器（Ea）；False -> 直接用缓存的 CLIP E
TEXT_ADAPTER_HIDDEN = 512      # 适配器隐层维度
TEXT_ADAPTER_GATE_INIT = 0.0   # 残差门控初值（0 = 训练初期恒等）
USE_ACM = True                 # 自适应跨模态掩码损失
ACM_LAMBDA = 0.1               # ACM 损失权重（前 ACM_WARMUP_EPOCHS 个 epoch 线性 warmup）
ACM_WARMUP_EPOCHS = 10         # 前 N 个 epoch 线性 warmup（0 -> ACM_LAMBDA）
ACM_MASK_RATIO = 0.15          # 文本元素级掩码比例
ACM_GRAD_TO_BACKBONE = True    # True -> 梯度回传到 MSDA 主干（让特征学会读出语义）
## =================================================================

## ============ 困难样本排空（保险机制） ============
## refinement 阶段（epoch > final_epoch_ratio）把剩余困难样本按 mean_ci 由高到低
## 逐步收进简单池，初始伪标签 = 点标注，之后由 FIU 自演化。对应 PAL 原版
## lose_point_ratio -> 1.0 “后期全部收下”的语义。
CI_DRAIN = True
CI_DRAIN_START_RATIO = 0.8     # 从总 epoch 的 80% 开始
CI_DRAIN_END_RATIO = 0.95      # 到 95% 排完
CI_DRAIN_START_FRAC = 0.2      # 每轮排空比例的起点（占剩余困难池）
## =================================================================

## 根据给定的choose_model, choose_dataset, choose_dataset_type生成对应文件夹作为工作空间,且加一个时间码确保文件夹的独一无二性,不存在覆盖问题 
MODEL_NAME = choose_model + '__' + choose_dataset + '__' + choose_dataset_type 
# + '__' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))


root_path = os.path.abspath('.')
# input_path = "./dataset/[choose_dataset]/"
input_path = os.path.join(root_path,'dataset',choose_dataset)
demo_train_input_path = os.path.join(input_path,MODEL_NAME)

# 默认使用的是linux格式的文件名
# 训练用的数据集用的是后来构造的，验证的数据集用的是原数据集里面的
TRAIN_IMG_DIR = demo_train_input_path + "/train/choose/img"
TRAIN_MASK_DIR = demo_train_input_path + "/train/choose/mask"
train_points_dir = demo_train_input_path + "/train/choose/points"

VAL_IMG_DIR = input_path + "/val/img"
VAL_MASK_DIR = input_path + "/val/mask"

nc_img_dir =  demo_train_input_path + "/train/no_choose/img"
nc_mask_dir =  demo_train_input_path + "/train/no_choose/mask_pred"
nc_points_dir = demo_train_input_path + "/train/no_choose/points"

num_images = len(os.listdir(VAL_MASK_DIR))

# 原数据集的路径
origin_dir = input_path + "/origin"
origin_img_dir = input_path + "/origin/img"
origin_points_dir = os.path.join(input_path,'origin',choose_dataset_type)

# PadImg函数将图像image的尺寸(H, W)，扩展到刚好大于(H,W)的指定times=32的最小倍数
def PadImg(image, times=32):
    h, w, c = image.shape
    pad_height = math.ceil(h / times) * times - h
    pad_width = math.ceil(w / times) * times - w
    image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
    return image


# TEST_PATCH_BATCH_SIZE 与 PATCH_SIZE_test 均为全局变量
# 该函数作用：输出模型的测试结果，并且为了避免占用显存过大，确保输入到net的img的最大尺寸为PATCH_SIZE_test（1024）。
# 注意：
# 1. 当批次为16的1k图像为200MB，每一张图像对应的中间层的大小会随着图像尺寸增大而增大
# 2. 当stride与patch_size相等时，而预测的结果刚好在两个分块之间，模型往往表现不佳；此时stride应该取patch_size//2，且对应的预测结果也应该有相应处理
def test_pred(img, net, batch_size = TEST_PATCH_BATCH_SIZE, choose_model = choose_model):
    b, c, h, w = img.shape
    #print(img.shape)
    patch_size = PATCH_SIZE_test
    stride = PATCH_SIZE_test

    
    if h > patch_size and w > patch_size:
        # unfold函数： (b,c,h,w) --> (b, c*patch_size*patch_size, L)
        img_unfold = F.unfold(img, kernel_size=patch_size, stride=stride)
        img_unfold = img_unfold.reshape(b, c, patch_size, patch_size, -1).permute(0, 4, 1, 2, 3)
        # patch_num == H/patch_sz * W/patch_sz == L
        patch_num = img_unfold.size(1)

        preds_list = []
        # 以下循环的作用：将img_unfold按照batch_size为一个批次放入net中预测，按照网络的模型将预测结果放入preds_list中
        for i in range(0, patch_num, batch_size):
            end = min(i + batch_size, patch_num)            # end不能越界,最大为patch_num
            # batch_patches.size == (b*batch_size, c, p, p)
            batch_patches = img_unfold[:, i:end, :, :, :].reshape(-1, c, patch_size, patch_size)
            batch_patches = Variable(batch_patches.float())         # Variable函数已淘汰，现在只需要确保batch_patches为tensor, 且属性requires_grad=true即可
            batch_preds = net.forward(batch_patches)
            if choose_model == 'DNA':
                preds_list.append(batch_preds[-1])
            elif choose_model == 'UIU':
                preds_list.append(batch_preds[0])
            else:
                preds_list.append(batch_preds)
        # cat函数： (b*batch_size,c,p,p) --> (b*(alpha*batch_size)=patch_num,c,p,p)
        preds_unfold = torch.cat(preds_list, dim=0).permute(1, 2, 3, 0)
        preds_unfold = preds_unfold.reshape(b, -1, patch_num)
        #preds_unfold.sz == (b, c*p*p, patch_num)
        preds = F.fold(preds_unfold, kernel_size=patch_size, stride=stride, output_size=(h, w))

    else:
        preds = net.forward(img)
        if choose_model == 'DNA':
            preds = preds[-1]
        elif choose_model == 'UIU':
            preds = preds[0]
        else:
            pass
    return preds


class SirstDataset_test(Dataset):
    def __init__(self, image_dir, mask_dir, img_name_list, transform=None, mode='None'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_list = img_name_list
        self.mode = mode

    def __len__(self):
        return len(self.image_list)

    # for aug_image, image_name, h, w in dataloader: 中会使用到该函数返回的参数
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image_list[index])
        mask_path = os.path.join(self.mask_dir, self.image_list[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127.5).astype(float)
        # 新伪标签与原图同尺寸；只兼容旧版 256×256 标签。
        mask = match_mask_to_image(mask, image)

        if (self.mode == 'test'):
            # 只 pad 到 32 的倍数，不 resize；返回原图 h,w 用于把预测裁回原坐标。
            image, _, h, w = pad_to_multiple(image, mask=None, multiple=32)
            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]
            return image, mask, self.image_list[index], h, w

        else:
            print("mode输入的格式不对")
            sys.exit(0)



def main():
    # origin_img_dir = "./dataset/[choose_dataset]/origin/img"
    # Calculate_meas_std函数作用：计算选中的数据集所有图像的mean, std, 并打印
    # 通常彩色图像的均值为0.45, 方差为0.22
    # cal_mean, cal_std = Calculate_mean_std(origin_img_dir)
    cal_mean = float(0.38625179481616323)
    cal_std = float(0.14124445249304624)

    def train_fn(loader, model, optimizer, loss_fn, scaler, epoch, choose_model=choose_model,
                 acm=None, acm_lambda=0.0, grad_clip=None):
        model.train()
        # ACM 只在 λ>0 时才捕获特征（关闭时 pre-hook 直接返回，不占显存）
        if acm is not None:
            acm.set_enabled(acm_lambda > 0.0)
        acm_losses = []
        stat_buffers = {}       # 损失函数自报的统计量（mean_p_pos / pred_fg_px ...）
        # 当ncols = 0不会显示进度条，仅显示一些关键信息。tqdm类不会影响loader的正常使用，仅仅在loader被请求新值更新进度条
        loop = tqdm(loader, ncols=fixed_length)

        # 这下面两个应该是用来计算metric的类
        iou_metric.reset()
        nIoU_metric.reset()

        train_losses = []
        # 一个epoch的训练 
        for batch_idx, (data, targets, edge) in enumerate(loop):
            data = data.to(device=DEVICE)
            # unsqueeze(1)函数作用：在targets的size[1]位置插入一个维度，e.g, targets.size=(4,2) --> (4,1,2),用于与模型输出对齐
            targets = targets.unsqueeze(1).to(device=DEVICE)
            edge = edge.to(device=DEVICE)

            if choose_model == 'DNA':
                # 自动混合精度（AMP, Automatic Mixed Precision）训练的启动开关，在前向传播时使用低精度，在计算损失时使用高精度;
                # 直接用模型的原始输出计算损失好处：数值稳定性和更高的计算与内存效率;
                # 关注一下loss_fn与loader类是如何计算损失的！
                with torch.cuda.amp.autocast():
                    predictions_no_sigmoid = model(data)
                    loss_0 = loss_fn(predictions_no_sigmoid[0], targets, edge)
                    loss_1 = loss_fn(predictions_no_sigmoid[1], targets, edge)
                    loss_2 = loss_fn(predictions_no_sigmoid[2], targets, edge)
                    loss_3 = loss_fn(predictions_no_sigmoid[3], targets, edge)
                    loss = torch.mean(loss_0 + loss_1 + loss_2 + loss_3)
                predictions = torch.sigmoid(predictions_no_sigmoid[-1])

            elif choose_model == 'UIU':
                with torch.cuda.amp.autocast():
                    predictions_no_sigmoid = model(data)
                    loss_0 = loss_fn(predictions_no_sigmoid[0], targets, edge)
                    loss_1 = loss_fn(predictions_no_sigmoid[1], targets, edge)
                    loss_2 = loss_fn(predictions_no_sigmoid[2], targets, edge)
                    loss_3 = loss_fn(predictions_no_sigmoid[3], targets, edge)
                    loss_4 = loss_fn(predictions_no_sigmoid[4], targets, edge)
                    loss_5 = loss_fn(predictions_no_sigmoid[5], targets, edge)
                    loss_6 = loss_fn(predictions_no_sigmoid[6], targets, edge)
                    loss = torch.mean(loss_0 + loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6)
                predictions = torch.sigmoid(predictions_no_sigmoid[0])
            # 其他模型
            else:
                with torch.cuda.amp.autocast():
                    predictions_no_sigmoid = model(data)
                    loss = loss_fn(predictions_no_sigmoid, targets, edge)
                predictions = torch.sigmoid(predictions_no_sigmoid)

            iou_metric.update(predictions, targets)
            nIoU_metric.update(predictions, targets)
            _, IoU = iou_metric.get()
            _, nIoU = nIoU_metric.get()

            # ---- ACM：用融合前的视觉特征重建被掩码的文本语义 ----
            # 注意：必须在 forward 之后、backward 之前取，且只对 loss 求一次。
            if acm is not None and acm_lambda > 0.0:
                l_acm = acm.forward_loss()
                if l_acm is not None:
                    loss = loss + acm_lambda * l_acm
                    acm_losses.append(float(l_acm.detach()))

            # 混合精度的后向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            train_losses.append(loss.item())
            # 高斯 focal loss 量级可能偏大，做一次梯度裁剪（AMP 下必须先 unscale）
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()     # 根据本轮训练数据是否出现溢出，来动态调整放大的倍数因子

            # 收集本 batch 的损失统计（GaussianFocalLoss 才有 last_stats）
            _st = getattr(loss_fn, 'last_stats', None)
            if _st:
                for _k, _v in _st.items():
                    stat_buffers.setdefault(_k, []).append(float(_v))

            ''' 正常的后向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            '''

            ''' 余弦退火来动态调整学习率
            scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=0)
            每一个epoch后scheduler.step()
            其他一切如上操作
            '''

            # update tqdm loop
            loop.set_description(f"train epoch is：{epoch + 1} ")
            loop.set_postfix(loss=loss.item(), IoU=IoU.item(), nIoU=nIoU.item())

        # SigmoidMetric.get() 返回的 IoU 是 shape=(1,) 的 numpy.ndarray，
        # 直接传入内置 round() 会报 "type numpy.ndarray doesn't define __round__ method"，
        # 这里统一转成 python float，避免后续打印/比较出错。
        return float(IoU), float(nIoU), float(np.mean(train_losses)), \
               (float(np.mean(acm_losses)) if acm_losses else float('nan')), \
               {k: float(np.mean(v)) for k, v in stat_buffers.items()}

    # loss_fn = Edge_loss
    def val_fn(loader, model, loss_fn, epoch):
        model.eval()
        loop = tqdm(loader, ncols=fixed_length)
        iou_metric.reset()
        nIoU_metric.reset()
        FA_PD_metric.reset()
        eval_losses = []
        with torch.no_grad():
            for batch_idx, (x, y, h, w) in enumerate(loop):
                x = x.to(device=DEVICE)
                y = y.unsqueeze(1).to(device=DEVICE)
                h_i = int(h[0])
                w_i = int(w[0])
                preds_no_sigmoid = test_pred(x, model)
                # 预测和 GT 都裁回原图坐标；不做 resize/interpolate。
                preds_no_sigmoid = preds_no_sigmoid[:, :, :h_i, :w_i]
                y = y[:, :, :h_i, :w_i]
                preds = torch.sigmoid(preds_no_sigmoid)
                # 原代码 `eval_losses = loss_fn(preds, y, None)` 有两个问题：
                #   1) 此处 preds 尚未定义 -> NameError；2) edgeSCE_loss 需要 edge，传 None 会报错。
                # 验证阶段只用 IoU/nIoU/FA/PD，不算 loss。
                eval_losses = [0.0]
                iou_metric.update(preds, y)
                nIoU_metric.update(preds, y)
                FA_PD_metric.update(preds, y)
                _, IoU = iou_metric.get()
                _, nIoU = nIoU_metric.get()
                FA, PD = FA_PD_metric.get(num_images)
                loop.set_description(f"test epoch is：{epoch + 1} ")
                loop.set_postfix(IoU=IoU.item(), nIoU=nIoU.item())
        # 同上：IoU 为 ndarray，统一转成 python float
        return float(IoU), float(nIoU), float(np.mean(eval_losses)), float(FA), float(PD)

    train_transform = A.Compose(
        [
            # someof函数以0.5的概率被执行，执行时候随机从列表中选择并执行3个变换
            A.SomeOf([
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Transpose(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightness(limit=0.3, p=0.2),
                A.RandomBrightnessContrast(contrast_limit=0.3, p=0.2),
                A.Rotate(limit=45, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.5),
                A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=0, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.2),   
                A.NoOp(),   #无操作
                A.NoOp(),
            ], 3, p=0.5),
            A.Normalize(
                mean=cal_mean,
                std=cal_std,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    # 评估/伪标签更新只做 Normalize；几何上只 pad 到 32 的倍数，之后裁回原图。
    val_transforms = A.Compose(
        [
            A.Normalize(
                mean=cal_mean,
                std=cal_std,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    if choose_model == 'DNA' or choose_model == 'UIU':
        model = model_func(mode='train').to(DEVICE)
    else:
        model = model_func().to(DEVICE)

    # ==================== 文本先验(TATE) + CI 门控接入 ====================
    # 说明：必须在 optimizer 之前接入，且以 model.add_module 注册，
    #      这样下面 AdamW(model.parameters()) 才能自动包含文本模块的参数。
    # CI 门控与 TATE 共用同一份 CLIP 文本嵌入 E（若 TEXT_PRIOR_RANDOM=True，
    # 门控也使用随机文本，这正是“文本到底有没有信息量”消融所需的语义）。
    _embedder = None
    ci_scorer = None
    text_adapter = None
    acm = None
    if USE_TEXT_PRIOR or USE_CI_FILTER:
        _embedder = ClipTextEmbedder(
            SINGLE_FRAME_IR_PROMPTS,
            model_name_or_path=TEXT_CLIP_MODEL,
            device=DEVICE,
            # CI 门控需要 CLIP 图像塔；只做 TATE 时保持原行为（编码完即释放）
            keep_image_tower=USE_CI_FILTER,
        )
        text_embedding = _embedder.get()          # [N, 512]

        if TEXT_PRIOR_RANDOM:
            # 消融用：同形状随机向量，参数量/结构与真实文本完全一致
            _g = torch.Generator().manual_seed(0)
            text_embedding = torch.randn(text_embedding.shape, generator=_g)
            print('[TextPrior] **消融模式**：使用随机文本嵌入（random text embedding）')

        # ---- 文本适配器：Ea = E + gate*MLP(E)，gate 零初始化 ----
        # 只注册一次（挂在 model 上），TATE 与 ACM 共享同一实例，
        # 避免 state_dict 出现重名副本、同时保证 optimizer 能拿到它的参数。
        if USE_TEXT_ADAPTER:
            text_adapter = TextAdapter(text_embedding.size(1),
                                       hidden=TEXT_ADAPTER_HIDDEN,
                                       gate_init=TEXT_ADAPTER_GATE_INIT)
            try:
                text_adapter = text_adapter.to(next(model.parameters()).device)
            except StopIteration:
                pass
            model.add_module('text_adapter', text_adapter)
            n_ad = sum(p.numel() for p in text_adapter.parameters())
            print(f'[TextAdapter] Ea = E + gate*MLP(E)，gate_init={TEXT_ADAPTER_GATE_INIT}，'
                  f'参数量={n_ad} (~{n_ad/1e3:.1f}K)')

        if USE_TEXT_PRIOR:
            attach_text_prior(
                model,
                text_embedding,
                choose_model=choose_model,
                hidden_dim=TEXT_PRIOR_HIDDEN,
                use_gate=TEXT_PRIOR_USE_GATE,
                gate_init=TEXT_PRIOR_GATE_INIT,
                text_adapter=text_adapter,
            )

        if USE_CI_FILTER:
            ci_scorer = CIPatchScorer(
                _embedder.get_image_tower(),
                text_embedding,
                DEVICE,
                side_ratio=CI_CROP_SIDE_RATIO,
                min_side=CI_CROP_MIN,
                max_side=CI_CROP_MAX,
            )
            print('[CI-Filter] 已启用 CI 门控：S=概率图组件置信度，K=CLIP 图文余弦相似度，'
                  'theta=图级均值 CI 的 GMM 高簇峰值（跨轮 EMA 平滑）')

        if USE_ACM:
            acm, _acm_hook = attach_cross_modal(
                model, text_embedding, adapter=text_adapter,
                choose_model=choose_model,
                mask_ratio=ACM_MASK_RATIO,
                grad_to_backbone=ACM_GRAD_TO_BACKBONE,
            )
            print(f'[ACM] λ={ACM_LAMBDA}（warmup {ACM_WARMUP_EPOCHS} epoch），'
                  f'损失只算 attn·V 对未掩码文本的重建，无残差旁路')
    # =========================================================

    # ---- 损失函数选择：高斯 focal（默认） or 原 PAL 的 edgeSCE ----
    if LOSS_TYPE == 'gauss_focal':
        loss_fn = GaussianFocalLoss(alpha=GF_ALPHA, gamma=GF_GAMMA,
                                    pos_weight=GF_POS_WEIGHT, norm=GF_NORM,
                                    target=GF_TARGET, pos_eps=GF_POS_EPS,
                                    skip_empty=GF_SKIP_EMPTY)
        print(f'[Loss] GaussianFocal: α={GF_ALPHA} γ={GF_GAMMA} W={GF_POS_WEIGHT} '
              f'norm={GF_NORM} target={GF_TARGET} σ={GF_SIGMA_PX}({GF_SIGMA_MODE}) '
              f'skip_empty={GF_SKIP_EMPTY}')
    else:
        loss_fn = edgeSCE_loss
        print('[Loss] 使用原 PAL edgeSCE_loss')
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    iou_metric = SigmoidMetric()
    nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    FA_PD_metric = PD_FA_2(1)

    scaler = torch.cuda.amp.GradScaler()

    best_mIoU = 0
    best_nIoU = 0
    best_PD = 0
    bestmIoUandPD = 0
    best_mIoU_nIoU = 0
    best_mIoU_FA = 0
    best_mIoU_PD = 0
    best_mIoU_epoch = 0
    num_epoch = []
    num_train_loss = []
    num_test_loss = []
    num_mioU = []
    num_nioU = []

    # 只记录miou最佳时候的模型权重与信息
    save_model_file_path = os.path.join(root_path, 'work_dirs', MODEL_NAME)
    make_dir(save_model_file_path)
    save_file_name = os.path.join(save_model_file_path, MODEL_NAME + '.txt')
    save_best_miou_file_name = os.path.join(save_model_file_path,
                                            'best_mIoU_checkpoint_' + MODEL_NAME + ".pth.tar")
    # 第一阶段结束（即将进入 enhancement）时的权重文件，供 --resume1 使用
    stage1_end_epoch = int(NUM_EPOCHS * clear_inital_ratio)     # 最后一个 pre-start epoch
    resume1_start_epoch = stage1_end_epoch + 1                  # 第一个 enhancement epoch
    save_stage1_file_name = os.path.join(save_model_file_path,
                                         'stage1_checkpoint_' + MODEL_NAME + ".pth.tar")

    # ---- CI 门控的状态：阈值 EMA 与逐轮日志 ----
    # theta 的 EMA 需要跨轮保持；落盘是为了 --resume1 / 中断重启后阈值不重置。
    ci_theta_file = os.path.join(save_model_file_path, 'ci_theta.json')
    ci_log_dir = os.path.join(save_model_file_path, 'ci_log')
    ci_theta_ema = None
    ci_mean_by_name = {}      # 最近一轮的图级均值 CI，供 refinement 阶段的排空排序用
    make_dir(ci_log_dir)

    # 命令行参数：python train_model.py --resume1
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume1', action='store_true',
                        help='加载第一阶段结束的权重，直接从 epoch > int(NUM_EPOCHS*clear_inital_ratio) 继续训练')
    args = parser.parse_args()
    RESUME1 = args.resume1

    save_file = open(save_file_name, 'a')

    start_epoch = 0

    if RESUME1:
        # 没有第一阶段权重就直接报错，避免用随机权重“假装”续训
        if not os.path.exists(save_stage1_file_name):
            raise FileNotFoundError(
                f"--resume1 需要第一阶段权重文件，但未找到：{save_stage1_file_name}\n"
                f"请先正常运行一次 train_model.py，训练到 epoch={resume1_start_epoch} 时会自动保存该文件。")
        checkpoint = torch.load(save_stage1_file_name, map_location=DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']        # = resume1_start_epoch（即 enhancement 第一轮）
        best_mIoU = checkpoint.get('best_mIoU', 0)
        best_nIoU = checkpoint.get('best_nIoU', 0)
        best_PD = checkpoint.get('best_PD', 0)
        bestmIoUandPD = checkpoint.get('bestmIoUandPD', 0)
        print(f"[resume1] 已加载第一阶段权重：{save_stage1_file_name}，从 epoch={start_epoch} 开始训练")
        save_file.write(f"[resume1] 加载第一阶段权重，从 epoch={start_epoch + 1} 开始训练。\n")
        # 续训时恢复 CI 阈值 EMA，避免阈值从零重建造成第一轮放行数突变
        if USE_CI_FILTER and os.path.exists(ci_theta_file):
            try:
                with open(ci_theta_file, 'r') as f:
                    ci_theta_ema = float(json.load(f).get('theta_ema'))
                print(f"[CI-Filter] 恢复阈值 EMA：theta_ema={ci_theta_ema:.5f}")
                save_file.write(f"[CI-Filter] 恢复阈值 EMA：theta_ema={ci_theta_ema:.5f}\n")
            except Exception as e:
                print(f"[CI-Filter] 阈值 EMA 恢复失败（忽略）：{e}")
        save_file.flush()
    else:
        start_epoch = 0

    save_file.write(
        '\n---------------------------------------start--------------------------------------------------\n')
    save_file.write(datetime.now().strftime("%Y-%m-%d, %H:%M:%S\n"))

    # and后面的语句是我后期加上，避免重复创建训练集。
    # ⚠️ 判据必须是"池子里有没有样本"，而不是"目录存不存在"：
    #    __main__ 里会先用 make_dir() 建好 6 个空目录，若按目录存在与否判断，
    #    首次运行会跳过初始池生成 → choose/no_choose 都是 0 张 →
    #    get_loaders 报 num_samples=0（cv2.imwrite 对不存在的目录是静默失败）。
    if choose_dataset_type == 'masks_coarse' or choose_dataset_type == 'masks_centroid':
        _choose_empty = (not os.path.exists(TRAIN_IMG_DIR)) or len(os.listdir(TRAIN_IMG_DIR)) == 0
        _no_choose_empty = (not os.path.exists(nc_img_dir)) or len(os.listdir(nc_img_dir)) == 0

        if _choose_empty and _no_choose_empty:
            print("训练池为空，开始生成初始伪标签池（高斯+Canny，确定性，无随机）...")
            data_inital_make_add_points(origin_img_dir, origin_points_dir, TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                                        train_points_dir, nc_img_dir,
                                        nc_mask_dir, nc_points_dir, crop_size=10)  # 10*10区域
        elif _choose_empty != _no_choose_empty:
            # 一边空一边非空：说明池子被外部打断/手工删过，继续跑会让 A/B 不可比
            raise RuntimeError(
                f"训练池状态不一致（choose空={_choose_empty}, no_choose空={_no_choose_empty}）。\n"
                f"  请检查并二选一：\n"
                f"    (a) 换一个新的 MODEL_NAME 重新生成（推荐，A/B 起点才一致）\n"
                f"    (b) 同时删除以下两个目录后重跑：\n"
                f"        {demo_train_input_path}/train/choose\n"
                f"        {demo_train_input_path}/train/no_choose")

        # 生成/复用后强校验：避免空池跑到 epoch1 才在 DataLoader 里崩
        if len(os.listdir(TRAIN_IMG_DIR)) == 0:
            raise RuntimeError(
                f"初始训练池生成后仍为 0 张。请检查：\n"
                f"  图像目录: {origin_img_dir}\n"
                f"  点标注目录: {origin_points_dir}\n"
                f"  两者文件名是否一一对应、mask 是否为 0/255 点标注。")
        print(f"初始池就绪：choose={len(os.listdir(TRAIN_IMG_DIR))} 张，"
              f"no_choose={len(os.listdir(nc_img_dir))} 张")
    else:
        pass


    for epoch in range(start_epoch, NUM_EPOCHS):

        # !!!!  原代码此处硬编码 epoch = 100（调试遗留），会强制停在 enhancement 段
        # 并导致每轮都做伪标签更新。正式训练必须注释掉。
        # epoch = 100
        # !!!!

        # 首次进入 enhancement 阶段（epoch > int(NUM_EPOCHS*clear_inital_ratio)）时，
        # 保存此时（第一阶段刚结束）的模型权重，供 --resume1 使用。
        # 注意：必须在下面 enhancement 块的伪标签自更新之前保存，才是纯第一阶段权重。
        if epoch == resume1_start_epoch and not RESUME1:
            torch.save({
                'epoch': epoch,                 # 下次从该 epoch 开始（enhancement 第一轮）
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mIoU': best_mIoU,
                'best_nIoU': best_nIoU,
                'best_PD': best_PD,
                'bestmIoUandPD': bestmIoUandPD,
            }, save_stage1_file_name)
            print(f"已保存第一阶段权重：{save_stage1_file_name}")
            save_file.write(f"已保存第一阶段权重（epoch={epoch}）：{save_stage1_file_name}\n")
            save_file.flush()

        print(epoch)
        if choose_dataset_type == 'mask':
            train_loader, val_loader = get_loaders(  # 永远不会用
                origin_img_dir,
                origin_points_dir,
                VAL_IMG_DIR,
                VAL_MASK_DIR,
                PATCH_SIZE,
                TRAIN_BATCH_SIZE,
                TEST_BATCH_SIZE,
                train_transform,
                val_transforms,
                NUM_WORKERS,
                PIN_MEMORY,
            )

        elif choose_dataset_type == 'masks_coarse' or choose_dataset_type == 'masks_centroid':
            train_ds = SirstDataset_test(
                image_dir=TRAIN_IMG_DIR,
                mask_dir= TRAIN_MASK_DIR,
                img_name_list=os.listdir(TRAIN_IMG_DIR),
                transform=val_transforms,
                mode='test'
            )

            # 更新阶段保持原图尺寸，尺寸不一，只能 batch=1。
            train_loader_label_update = DataLoader(
                train_ds,
                batch_size=1,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                shuffle=False,
            )

            test_ds = SirstDataset_test(
                image_dir=nc_img_dir,
                # mask_dir= nc_mask_dir,
                mask_dir = nc_points_dir,
                img_name_list=os.listdir(nc_img_dir),
                transform=val_transforms,
                mode='test'
            )

            # 困难样本预测同样保持原图尺寸，batch=1。
            test_loader = DataLoader(
                test_ds,
                batch_size=1,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                shuffle=False,
            )

            #clear_initial_ratio = 0.2, final_epoch_ratio = 0.8, NUM_EPOCHS=400， lose_point_ratio_init（漏检测率阈值）=0.2
            if (epoch > int(NUM_EPOCHS * clear_inital_ratio) and epoch <= int(NUM_EPOCHS * final_epoch_ratio)):

                model.eval()
                if epoch % clear_epoch_gap != 0:
                    pass
                else:
                    # lose_point_ratio应该是随着epoch不断增加的
                    lose_point_ratio = lose_point_ratio_init + (epoch - (NUM_EPOCHS * clear_inital_ratio) + 1) / (NUM_EPOCHS * (final_epoch_ratio - clear_inital_ratio)) * (1 - lose_point_ratio_init)
                    
                    print("开始标签的自更新--------")
                    # data的来源是choose文件夹（容易样本）
                    for idx, (img, masks, name, h, w) in tqdm(enumerate(train_loader_label_update)):
                        img = img.to(device=DEVICE)
                        with torch.no_grad():
                            output = test_pred(img, model)  #type(output) = torch.Tensor
                            output = torch.sigmoid(output)
                            output = output.cpu().data.numpy()
                        # 对每一个掩码都做一次自更新,
                        for i in range(output.shape[0]):
                            h_i, w_i = int(h[i]), int(w[i])
                            pred = np.array(output[i, 0, :h_i, :w_i], dtype='float32')
                            prev_label = masks[i].cpu().data.numpy()
                            # 自更新函数；pred/mask 都在原图坐标，不 resize。
                            current_binary_pred = update_gt_update_degen_corr(
                                pred, prev_label, thresh_Tb, thresh_k, [h_i, w_i], degen=degen)
                            current_binary_pred = current_binary_pred * 255
                            cv2.imwrite(os.path.join(TRAIN_MASK_DIR, name[i]), current_binary_pred)

                    print("开始认识并学习困难样本-----------")
                     # test_loader用的是no_choose数据集，即困难样本  
                    if USE_CI_FILTER:
                        # ==================== CI 门控（TAPF 风格） ====================
                        # 替换 PAL 原判据：不再用"点位漏检率 + 虚警比"人工阈值，而是对每张
                        # 困难样本算 图级均值 CI（= 各点对应过点组件 CI 的均值，未匹配点记 0），
                        # 再对全体均值拟合 GMM(2)、取高簇密度峰值作为 theta；
                        # mean_ci >= theta 才进简单池。公式与推导见 components/credibility.py 头部。
                        # 注意：均值天然包含"漏点惩罚"，因此不再需要 lose_point_ratio 爬坡。
                        ci_records = []     # 通过虚警预筛、参与 GMM 的样本
                        ci_skipped = 0      # 被虚警预筛挡下的样本数（留在困难池）
                        auto_in = []        # 无点标注的样本：沿用原逻辑直接进简单池

                        # K 跟着文本适配器走（同一轮内固定，避免阈值与排序不同源）
                        if ci_scorer is not None:
                            _E_ci = text_embedding.to(DEVICE)
                            if text_adapter is not None:
                                with torch.no_grad():
                                    _E_ci = text_adapter(_E_ci)
                            ci_scorer.set_text_features(_E_ci)

                        for idx, (img, _, name, h, w) in tqdm(enumerate(test_loader)):
                            img = img.to(device=DEVICE)
                            with torch.no_grad():
                                output = test_pred(img, model)
                                output = torch.sigmoid(output)
                                output = output.cpu().data.numpy()

                            for i in range(output.shape[0]):
                                h_i, w_i = int(h[i]), int(w[i])
                                name_i = name[i]
                                # 原图坐标下的概率图，不 resize（S 与二值化都用它）
                                prob = np.array(output[i, 0, :h_i, :w_i], dtype='float32')
                                # CI 需要对原始灰度像素做 CLIP 裁剪，直接读盘（与原流程一致，无归一化污染）
                                gray = cv2.imread(os.path.join(nc_img_dir, name_i), cv2.IMREAD_GRAYSCALE)
                                points = cv2.imread(os.path.join(nc_points_dir, name_i), cv2.IMREAD_GRAYSCALE)
                                if gray is None or points is None:
                                    print(f"图像或点标注读取失败：{name_i}")
                                    continue

                                # 阶段 A：连通域 + 过点标记（纯几何，不消耗 CLIP）
                                comp = extract_components(prob, points, prob_thr=0.5)
                                if comp['n_points'] == 0:
                                    # 与原版一致：没有点标注的困难样本视为容易样本
                                    auto_in.append((name_i, comp['bin_mask']))
                                    continue
                                # 虚警预筛：不过点组件过多 -> 本轮不学，留在困难池
                                # 分母固定为点数，语义比原版 "除以预测组件数" 更稳
                                if spurious_ratio(comp['n_spurious'], comp['n_points']) > alarm_point_ration:
                                    ci_skipped += 1
                                    continue

                                # 阶段 B：过点组件的 CI（只有这一步用 CLIP 图像塔）
                                ci = sample_mean_ci(comp, gray, prob, ci_scorer)
                                ci_records.append(dict(
                                    name=name_i, mean_ci=ci['mean_ci'],
                                    ci_per_point=ci['ci_per_point'],
                                    S=ci['S'], K=ci['K'],
                                    n_points=comp['n_points'],
                                    n_spurious=comp['n_spurious'],
                                    bin_mask=comp['bin_mask'],
                                ))

                        # 阶段 C：GMM 动态阈值（统计单元 = 图级均值 CI），跨轮 EMA 平滑
                        pool = [r['mean_ci'] for r in ci_records]
                        theta_gmm = gmm_threshold(pool, min_samples=CI_MIN_POOL)
                        if theta_gmm is None:
                            # 样本太少无法稳定拟合 2 分量：回退池中位数（约放行一半）
                            theta_gmm = fallback_threshold(pool, CI_FALLBACK)
                        if theta_gmm > 0.0:
                            if ci_theta_ema is None:
                                ci_theta_ema = theta_gmm
                            else:
                                ci_theta_ema = CI_EMA_BETA * ci_theta_ema + (1.0 - CI_EMA_BETA) * theta_gmm
                            theta = ci_theta_ema
                        else:
                            # 全部 mean_ci 都为 0（所有点都没被检出）：本轮不放行。
                            # 不能把这个 0 写进 EMA，否则后续阈值会被永久拉成 0。
                            theta = float('inf')

                        # 阶段 D：放行（theta 决定"够不够格"，k 决定"这轮最多进几个"）
                        passed = [r for r in ci_records if r['mean_ci'] >= theta]
                        # 记录本轮 mean_ci，供 refinement 阶段的排空排序使用
                        for r in ci_records:
                            ci_mean_by_name[r['name']] = r['mean_ci']
                        if CI_USE_QUOTA:
                            # 数量闸：k = 放行比例 × 当前困难池总数；比例随阶段从
                            # CI_ADMIT_RATIO_INIT 升到 1.0（斜率同 PAL 的 lose_point_ratio）。
                            # 池子数在搬文件之前取，才是本轮真实的基数。
                            n_pool = len(os.listdir(nc_img_dir))
                            ratio = admit_ratio(epoch, NUM_EPOCHS, CI_ADMIT_RATIO_INIT,
                                                clear_inital_ratio, final_epoch_ratio)
                            k = max(1, int(round(ratio * n_pool)))
                            # 超过 k 的部分按 mean_ci 从高到低取，差额留在困难池下一轮再判
                            passed.sort(key=lambda x: x['mean_ci'], reverse=True)
                            accepted = passed[:k]
                            quota_cut = len(passed) - len(accepted)
                        else:
                            accepted, k, quota_cut, ratio = passed, -1, 0, 1.0
                        new_choose_list = [r['name'] for r in accepted] + [n for n, _ in auto_in]
                        # 落盘二值预测：后续 deal_gen_mask_error_aera 会只保留过点连通域
                        # 并与点标注融合（虚警在精炼时被剔除），文件名与原流程保持一致。
                        for r in accepted:
                            cv2.imwrite(os.path.join(nc_mask_dir, r['name']), r['bin_mask'] * 255)
                        for name_i, bin_mask in auto_in:
                            cv2.imwrite(os.path.join(nc_mask_dir, name_i), bin_mask * 255)

                        # ---- 日志：阈值 / 放行数 / 分布 / 按点数分组 / 逐样本明细 ----
                        if len(pool) > 0:
                            q = np.percentile(np.asarray(pool), [10, 50, 90])
                            mean_S = float(np.mean([np.mean(r['S']) for r in ci_records if len(r['S'])]))
                            mean_K = float(np.mean([np.mean(r['K']) for r in ci_records if len(r['K'])]))
                        else:
                            q = [0.0, 0.0, 0.0]
                            mean_S = mean_K = 0.0
                        ci_msg = (f"[CI-Filter] epoch:{epoch + 1} 候选:{len(ci_records)} "
                                  f"虚警预筛挡下:{ci_skipped} theta_gmm:{theta_gmm:.5f} "
                                  f"theta_ema:{ci_theta_ema if ci_theta_ema is not None else -1:.5f} "
                                  f"本轮theta:{theta:.5f} 过theta:{len(passed)} k:{k} "
                                  f"被k卡掉:{quota_cut} 放行:{len(accepted)}(+无点{len(auto_in)}) "
                                  f"放行比例:{ratio:.3f} "
                                  f"mean_ci p10/p50/p90:{q[0]:.5f}/{q[1]:.5f}/{q[2]:.5f} "
                                  f"avgS:{mean_S:.4f} avgK:{mean_K:.4f}")
                        print(ci_msg)
                        save_file.write(ci_msg + "\n")
                        # 单点图的 mean_ci 就是单个组件的值（噪声大），多点图被平均过；
                        # 分开统计才能判断多点图是否被系统性卡住。
                        for tag, arr in (('单点图', [r['mean_ci'] for r in ci_records if r['n_points'] == 1]),
                                         ('多点图', [r['mean_ci'] for r in ci_records if r['n_points'] > 1])):
                            if arr:
                                acc = sum(1 for v in arr if v >= theta)
                                save_file.write(f"  {tag}: n={len(arr)} 过theta={acc} "
                                                f"mean_ci均值={np.mean(arr):.5f}\n")
                        # 阈值落盘：供 --resume1 / 中断重启后恢复 EMA
                        try:
                            with open(ci_theta_file, 'w') as f:
                                json.dump(dict(epoch=epoch + 1, theta_gmm=theta_gmm,
                                               theta_ema=ci_theta_ema), f)
                        except Exception as e:
                            print(f"[CI-Filter] 阈值落盘失败（忽略）：{e}")
                        # 逐样本明细：用于事后分析 CI 分布、S/K 量级与放行原因
                        # admitted 记录"最终是否真的进简单池"（已包含数量闸的裁剪）
                        accepted_names = {r['name'] for r in accepted}
                        try:
                            with open(os.path.join(ci_log_dir, f"ci_round_epoch{epoch + 1:04d}.jsonl"), 'w') as f:
                                for r in ci_records:
                                    f.write(json.dumps(dict(
                                        name=r['name'], mean_ci=r['mean_ci'],
                                        ci_per_point=r['ci_per_point'], S=r['S'], K=r['K'],
                                        n_points=r['n_points'], n_spurious=r['n_spurious'],
                                        pass_theta=bool(r['mean_ci'] >= theta),
                                        admitted=bool(r['name'] in accepted_names),
                                    )) + "\n")
                        except Exception as e:
                            print(f"[CI-Filter] 明细落盘失败（忽略）：{e}")
                        save_file.flush()
                        # ==================== 以上为 CI 门控，以下与原流程一致 ====================

                    else:
                        # ---------------- 原版 PAL 门控（A/B 对照用，默认不启用） ----------------
                        for idx, (img, _, name, h, w) in tqdm(enumerate(test_loader)):
                            img = img.to(device=DEVICE)
                            with torch.no_grad():
                                output = test_pred(img, model)
                                output = torch.sigmoid(output)
                                output = output.cpu().data.numpy()

                            for i in range(output.shape[0]):
                                h_i, w_i = int(h[i]), int(w[i])
                                pred = np.array(output[i, 0, :h_i, :w_i], dtype='float32')
                                # 原图坐标下二值化，不 resize。
                                pred = np.where(pred > 0.5, 255, 0).astype(np.uint8)
                                cv2.imwrite(os.path.join(nc_mask_dir, name[i]), pred)

                        # deal_pred_mask_and_true_point_in 函数找到哪些困难样本可以被模型识别出来了并放在new_choose_list
                        new_choose_list = deal_pred_mask_and_true_point_in(nc_img_dir, nc_mask_dir,
                                                                           nc_points_dir, TRAIN_IMG_DIR,
                                                                           TRAIN_MASK_DIR, train_points_dir,
                                                                           lose_point_ratio=lose_point_ratio,
                                                                           alarm_point_ration=alarm_point_ration)

                    print(new_choose_list)
                    print(len(new_choose_list))
                    save_file.write(f"当前epoch:{epoch + 1}  进入张数:{len(new_choose_list)} \n")   # 第一次预测都有500多张可以进入，我怀疑模型有与训练权重
                    # deal_gen_mask_error_aera 函数在new_choose_list中，实现了根据点标注来剔除伪标签中错误的部分，并让为标签与点标注强行融合
                    deal_gen_mask_error_aera(nc_mask_dir, nc_points_dir, new_choose_list)

                   # hard_sample_in函数，将困难样本中new_choose_list的样本转移到简单样本的文件夹中
                    hard_sample_in(nc_img_dir, nc_mask_dir, nc_points_dir, TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                                       train_points_dir, new_choose_list)
                    print("此轮数据转移完成！！！！！")
                    # 记录本次迁移后的池子规模，便于观察演化
                    remaining_no_choose = len(os.listdir(nc_img_dir))
                    current_choose = len(os.listdir(TRAIN_IMG_DIR))
                    print(f"epoch:{epoch + 1}  no_choose剩余:{remaining_no_choose}  choose当前:{current_choose}")
                    save_file.write(
                        f"当前epoch:{epoch + 1}  进入张数:{len(new_choose_list)}  "
                        f"no_choose剩余:{remaining_no_choose}  choose当前:{current_choose}\n")
                    save_file.flush()

            elif epoch > int(NUM_EPOCHS * final_epoch_ratio):
                model.eval()
                if epoch % clear_epoch_gap != 0:
                    pass
                else:
                    for idx, (img, masks, name, h, w) in tqdm(enumerate(train_loader_label_update)):
                        img = img.to(device=DEVICE)
                        with torch.no_grad():
                            output = test_pred(img, model)
                            output = torch.sigmoid(output)
                            output = output.cpu().data.numpy()

                        for i in range(output.shape[0]):
                            h_i, w_i = int(h[i]), int(w[i])
                            pred = np.array(output[i, 0, :h_i, :w_i], dtype='float32')
                            prev_label = masks[i].cpu().data.numpy()
                            # 自更新；pred/mask 都在原图坐标，不 resize。
                            current_binary_pred = update_gt_update_degen_corr(
                                pred, prev_label, thresh_Tb, thresh_k, [h_i, w_i], degen=degen)
                            current_binary_pred = current_binary_pred * 255
                            cv2.imwrite(os.path.join(TRAIN_MASK_DIR, name[i]), current_binary_pred)

                    # ---- 保险机制：排空困难池 ----
                    # 背景：CI 门控在后期会让 mean_ci≡0 的样本永久停在困难池（实测 61/68 张
                    # 48 轮未进），而 PAL 原语义是 lose_point_ratio->1.0 后期全部收下。
                    # 这里在 refinement 阶段按 mean_ci 从高到低逐步收，初始伪标签 = 点标注，
                    # 之后由上面的 FIU 自演化。
                    if CI_DRAIN and ci_scorer is not None:
                        n_left = len(os.listdir(nc_img_dir))
                        if n_left > 0:
                            _start = int(NUM_EPOCHS * CI_DRAIN_START_RATIO)
                            _span = max(1, int(NUM_EPOCHS * (CI_DRAIN_END_RATIO - CI_DRAIN_START_RATIO)))
                            _prog = min(1.0, max(0.0, (epoch - _start + 1) / _span))
                            _frac = CI_DRAIN_START_FRAC + (1.0 - CI_DRAIN_START_FRAC) * _prog
                            _k = max(1, int(math.ceil(_frac * n_left)))
                            _names = sorted(os.listdir(nc_img_dir),
                                            key=lambda n: ci_mean_by_name.get(n, -1.0), reverse=True)
                            _pick = _names[:_k]
                            for _n in _pick:
                                _pts = cv2.imread(os.path.join(nc_points_dir, _n), cv2.IMREAD_GRAYSCALE)
                                if _pts is None:
                                    continue
                                # 初始伪标签只能是点（这些样本没有任何过点组件）
                                cv2.imwrite(os.path.join(nc_mask_dir, _n), _pts)
                            _moved = 0
                            for _n in _pick:
                                if not os.path.exists(os.path.join(nc_mask_dir, _n)):
                                    continue
                                move_files([_n], nc_img_dir, TRAIN_IMG_DIR)
                                move_files([_n], nc_mask_dir, TRAIN_MASK_DIR)
                                move_files([_n], nc_points_dir, train_points_dir)
                                _moved += 1
                            drain_msg = (f"[Drain] epoch:{epoch + 1} 剩余:{n_left} 本轮:{_moved} "
                                         f"比例:{_frac:.2f} 剩余后:{len(os.listdir(nc_img_dir))}")
                            print(drain_msg)
                            save_file.write(drain_msg + "\n")
                            save_file.flush()
            else:
                pass
            
            # 每个 epoch 都记录当前池子规模（即使本轮没做更新），便于观察 no_choose 演化
            choose_num = len(os.listdir(TRAIN_IMG_DIR))
            no_choose_num = len(os.listdir(nc_img_dir))
            print(f"epoch:{epoch + 1}  当前 choose 池:{choose_num}  no_choose 池:{no_choose_num}")
            save_file.write(f"当前epoch:{epoch + 1}  choose池总数:{choose_num}  no_choose池剩余:{no_choose_num}\n")
            save_file.flush()

            train_loader, val_loader = get_loaders(
                TRAIN_IMG_DIR,
                TRAIN_MASK_DIR,     # 使用简单样本的伪掩码来训练，伪掩码都是结合了点标签的信息
                VAL_IMG_DIR,
                VAL_MASK_DIR,       # 使用原始数据集中的验证集来验证
                PATCH_SIZE,
                TRAIN_BATCH_SIZE,
                TEST_BATCH_SIZE,
                train_transform,
                val_transforms,
                NUM_WORKERS,
                PIN_MEMORY,
                train_pointdir=train_points_dir,      # 高斯热图由点标注生成
                crop_fov=CROP_FOV,
                aux_mode=('heatmap' if LOSS_TYPE == 'gauss_focal' else 'edge'),
                gf_sigma=GF_SIGMA_PX,
                gf_sigma_mode=GF_SIGMA_MODE,
                gf_sigma_min=GF_SIGMA_MIN,
                gf_sigma_max=GF_SIGMA_MAX,
            )

        else:
            print("choose_dataset_type赋予的值不对，应该为[mask, masks_coarse, masks_centroid]中的一个")
            sys.exit(0)


        # ---- ACM 损失权重 warmup（0 -> ACM_LAMBDA）----
        if acm is not None and USE_ACM:
            acm_lambda = ACM_LAMBDA * min(1.0, (epoch + 1) / max(1, ACM_WARMUP_EPOCHS))
        else:
            acm_lambda = 0.0

        train_mioU, train_nioU, train_loss, train_acm_loss, gf_stats = train_fn(
            train_loader, model, optimizer, loss_fn, scaler, epoch,
            acm=acm, acm_lambda=acm_lambda, grad_clip=GF_GRAD_CLIP)

        # ---- 高斯 focal loss 的逐 epoch 统计 ----
        # task_loss 不包含 ACM 项，便于与旧 edgeSCE 直接对比；
        # mean_p_pos/mean_p_neg：正/负样本处的平均预测概率（学得好不好）；
        # pos_px：每张图的正样本像素数；pred_fg_px：预测前景面积中位数（防膨胀）；
        # empty_ratio：因“裁不到目标”被跳过的样本占比（GF_SKIP_EMPTY 生效时才有）。
        if gf_stats:
            gf_msg = (f"当前epoch:{epoch + 1}  "
                      f"task:{gf_stats.get('task_loss', float('nan')):.5f}  "
                      f"p_pos:{gf_stats.get('mean_p_pos', float('nan')):.4f}  "
                      f"p_neg:{gf_stats.get('mean_p_neg', float('nan')):.4f}  "
                      f"pos_px:{gf_stats.get('pos_px', 0):.1f}  "
                      f"pred_fg:{gf_stats.get('pred_fg_px', 0):.0f}  "
                      f"empty:{gf_stats.get('empty_ratio', 0):.3f}")
            print(gf_msg)
            save_file.write(gf_msg + "\n")
            save_file.flush()

        if USE_ACM and acm is not None and acm_lambda > 0:
            _st = getattr(acm, 'last_stats', {}) or {}
            acm_msg = (f"当前epoch:{epoch + 1}  acm_lambda:{acm_lambda:.4f}  "
                       f"acm_loss:{train_acm_loss:.5f}  "
                       f"masked_mse:{_st.get('masked', float('nan')):.5f}  "
                       f"kept_mse:{_st.get('unmasked', float('nan')):.5f}")
            print(acm_msg)
            save_file.write(acm_msg + "\n")
            if acm_lambda >= ACM_LAMBDA - 1e-9:
                # warmup 结束后记录门控值，用来判断模型到底“用了多少”文本
                _g = []
                if text_adapter is not None:
                    _g.append(f"adapter_gate={float(text_adapter.gate):.4f}")
                try:
                    _g.append(f"tate_alpha={float(model.text_prior.alpha):.4f}")
                except Exception:
                    pass
                if _g:
                    save_file.write(f"当前epoch:{epoch + 1}  " + "  ".join(_g) + "\n")
            save_file.flush()

        if epoch + 1 > num_start_test_epochs:   # 其实上面伪标签的更新的操作都可以挪到validation阶段来做,没必要再去重复用test_pred函数
            mioU, nioU, test_loss, fa, pd = val_fn(val_loader, model, loss_fn, epoch)

            # save model
            checkpoint = {
                'epoch': epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                'best_mIoU': best_mIoU,
                'best_nIoU': best_nIoU,
                'best_PD': best_PD,
                'bestmIoUandPD': bestmIoUandPD,
            }
            
            # 只记录最好的miou下的所有评价指标
            if best_mIoU < mioU:
                best_mIoU = mioU
                best_mIoU_nIoU = nioU
                best_mIoU_FA = fa
                best_mIoU_PD = pd
                best_mIoU_epoch = epoch
                torch.save(checkpoint, save_best_miou_file_name)
            else:
                best_mIoU = best_mIoU

            # 每次验证都要打印这么多东西，有点浪费资源
            print(f"当前epoch:{epoch + 1}  train_mioU:{round(train_mioU, 4)}  train_nioU:{round(train_nioU, 4)} \n"
                  f"当前epoch:{epoch + 1}  mioU:{round(mioU, 4)}  nioU:{round(nioU, 4)}  FA:{round(fa * 1000000, 3)}  PD:{round(pd, 4)} \n"
                  f"best_epoch:{best_mIoU_epoch + 1}  best_miou:{round(best_mIoU, 4)}  b_niou:{round(best_mIoU_nIoU, 4)}  FA:{round(best_mIoU_FA * 1000000, 3)}  PD:{round(best_mIoU_PD, 4)}\n"
                  )

            save_file.write(
                f"当前epoch:{epoch + 1}  train_mioU:{round(train_mioU, 4)}  train_nioU:{round(train_nioU, 4)} \n")
            save_file.write(
                f"epoch is:{epoch + 1}  mioU:{round(mioU, 4)}  nioU:{round(nioU, 4)}  FA:{round(fa, 3)}  PD:{round(pd, 4)}\n")
            save_file.write(
                f"best_epoch:{best_mIoU_epoch + 1}  best_miou:{round(best_mIoU, 4)}  b_niou:{round(best_mIoU_nIoU, 4)}  FA:{round(best_mIoU_FA * 1000000, 3)}  PD:{round(best_mIoU_PD, 4)}\n")

        else:
            mioU = 0
            nioU = 0
            test_loss = 100000
            fa = 100000
            pd = 0
            print(f"当前epoch:{epoch + 1}  train_mioU:{round(train_mioU, 4)}  train_nioU:{round(train_nioU, 4)}")
            save_file.write(
                f"当前epoch:{epoch + 1}  train_mioU:{round(train_mioU, 4)}  train_nioU:{round(train_nioU, 4)} \n")

        # 以下的num系列的列表，作者只修改，但都没使用
        num_epoch.append(epoch + 1)
        num_train_loss.append(train_loss)
        num_test_loss.append(test_loss)
        num_mioU.append(mioU)
        num_nioU.append(nioU)

    # epoch循环结束
    save_file.write(datetime.now().strftime("%Y-%m-%d, %H:%M:%S\n"))
    save_file.write('\n---------------------------------------end--------------------------------------------------\n')


# 只有使用命令python train_model.py时，__name__ == "__main__"等式才成立，否则为false。
if __name__ == "__main__":
    # 检查是否存在重复路径
    # check_path(demo_train_input_path)
    if choose_dataset_type == 'mask':
        pass
    elif choose_dataset_type == 'masks_coarse' or choose_dataset_type == 'masks_centroid':
        make_dir(TRAIN_IMG_DIR)
        make_dir(TRAIN_MASK_DIR)
        make_dir(train_points_dir)

        make_dir(nc_img_dir)
        make_dir(nc_mask_dir)
        make_dir(nc_points_dir)
    else:
        print("choose_dataset_type赋予的值不对，应该为[mask, masks_coarse, masks_centroid]中的一个")
        sys.exit(0)
    main()
