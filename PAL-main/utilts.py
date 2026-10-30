#!/usr/bin/python3
# coding = gbk
"""
@Author : yuchuang
@Time :
@desc:
"""

import os

import numpy as np

import cv2
import importlib
import random
import shutil

import sys
import torch
# import pydensecrf.densecrf as dcrf
# from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral

def make_dir(path):
    if os.path.exists(path)==False:
        os.makedirs(path)

def access_model(choose_model):
    choose_model_dir_name = choose_model + '_no_sigmoid'
    model_function = choose_model + '_No_Sigmoid'
    module_name = f"model.{choose_model}.{choose_model_dir_name}"  
    module = importlib.import_module(module_name)  
    model_func = getattr(module, model_function)  
    return model_func


def check_path(path):
    if os.path.exists(path) == True:
        print("Error: The workspace of the training pool already exists. Please manually and carefully delete the existing directory or add a suffix to the name of the created folder!!!")
        print("The conflicting directories are:", path)
        print("Method 3 is recommended to generate a unique training pool folder name!!!")
        sys.exit(0)


def center_point_inside_contour(center_point, target_mask):
    (center_y, center_x) = center_point     # 根据代码逻辑这里center_x与center_y都是数值
    # 在该函数中请记住，要查找的物体应该是白色，背景应该是黑色。最终得到各个连通区域的边缘轮廓点的列表，
    # RETR_EXTERNAL表示不考虑内部嵌套的情况，CHAIN_APPROX_SIMPLE表示不存储所有的轮廓点，而是删除所有冗余点并压缩轮廓
    target_contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    #center_index_XmergeY = {center_y * 1.0 + center_x * 0.0001,center_y-1 * 1.0 + center_x * 0.0001,center_y * 1.0 + center_x-1 * 0.0001,center_y+1 * 1.0 + center_x * 0.0001,center_y * 1.0 + center_x+1 * 0.0001}
    center_index_XmergeY = {center_y * 1.0 + center_x * 0.0001}     # 将二维坐标转化为一位坐标，其中y占整数位置，x占小数位置
    temp_contour_mask = np.zeros(target_mask.shape, np.uint8)

    overlap_found = False
    for target_contour in target_contours:
        target_contour_mask = np.zeros(target_mask.shape, np.uint8)
        cv2.fillPoly(target_contour_mask, [target_contour], (255))  # 根据target_contour的轮廓点对target_contour_mask填充
        target_index = np.where(target_contour_mask == 255)
        target_index_XmergeY = set(target_index[0] * 1.0 + target_index[1] * 0.0001)
        if not center_index_XmergeY.isdisjoint(target_index_XmergeY):   # isdisjoint()函数判断两个集合之间有没有交集，如果center_index_XmergeY为数值，等价于if center_index_XmergeY in target_index_XmergeY:
            area = cv2.contourArea(target_contour)
            if area>50:
                break
            else:
                overlap_found = True
                #print("True")
                cv2.fillPoly(temp_contour_mask, [target_contour], (255))
            break

    if not overlap_found:
        temp_contour_mask = temp_contour_mask
    return temp_contour_mask, overlap_found


def process_image(y_and_x, y1_and_y2_and_x1_and_x2, img_shape,image, low_threshold=50, high_threshold=150, kernel_size=(3, 3), sigma=0):
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    high_pass = cv2.subtract(image, blurred_image)
    edges = cv2.Canny(high_pass, low_threshold, high_threshold)
    kernel = np.ones((3, 3), np.uint8)
    sparse_edges = cv2.dilate(edges, kernel, iterations=1)
    sparse_edges = cv2.erode(sparse_edges, kernel, iterations=1)    # 最终的sparse_edges应该为二值边缘图,极端情况为单像素

    temp_contour_mask_2 = np.zeros(img_shape, np.uint8)
    (y1,y2, x1,x2) = y1_and_y2_and_x1_and_x2
    temp_contour_mask_2[y1:y2, x1:x2] = sparse_edges
    (y,x) = y_and_x

    # 以下函数功能是得到更精细的掩码 以及判断点标注是否在边界内，即判断是否传统图像处理方法与点标注在roi区域上是否吻合
    refine_mask,flag = center_point_inside_contour((y,x), temp_contour_mask_2)
    refine_mask_out = refine_mask[y1:y2, x1:x2]
    return refine_mask_out,flag



def data_inital_make_add_points(origin_img_dir, origin_points_dir,TRAIN_IMG_DIR,TRAIN_MASK_DIR,train_points_dir,nc_img_dir,nc_mask_dir,nc_points_dir, crop_size=10):
    input_image_path = origin_img_dir
    input_points_path = origin_points_dir

    # 防御：自己建好输出目录。否则 cv2.imwrite 会静默失败（不抛异常），
    # 导致"目录建了、文件没写进去"，下次再跑又因目录存在而跳过生成。
    for _d in (TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_points_dir,
               nc_img_dir, nc_mask_dir, nc_points_dir):
        make_dir(_d)

    # 从以上数据集来构建的训练池的位置（训练调度的环节）
    output_image_path = TRAIN_IMG_DIR       # ...train/choose/img
    output_masks_path = TRAIN_MASK_DIR      # ...train/choose/mask
    output_points_path = train_points_dir   # ...train/choose/points

    no_choose_output_image_path = nc_img_dir        # ...train/no_choose/img
    no_choose_output_masks_path = nc_mask_dir       # ...train/no_choose/mask_pred
    no_choose_output_points_path = nc_points_dir    # ...train/no_choose/points

    input_img_list = os.listdir(input_image_path)

    for i in range(len(input_img_list)):
        #print(f"正在处理图像：{input_img_list[i]}")
        img_path = os.path.join(input_image_path, input_img_list[i])
        points_path = os.path.join(input_points_path, input_img_list[i])

        out_img_path = os.path.join(output_image_path, input_img_list[i])
        out_mask_path = os.path.join(output_masks_path, input_img_list[i])
        out_points_path = os.path.join(output_points_path, input_img_list[i])

        no_choose_out_img_path = os.path.join(no_choose_output_image_path, input_img_list[i])
        no_choose_out_mask_path = os.path.join(no_choose_output_masks_path, input_img_list[i])
        no_choose_out_points_path = os.path.join(no_choose_output_points_path, input_img_list[i])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(points_path, cv2.IMREAD_GRAYSCALE)    # 得到的是(h,w,c)的numpy数组，通道顺序为BGR

        if img is None or mask is None:
            print(f"图像或mask读取失败：{input_img_list[i]}")
            continue

        # 得到mask二维数组中值为255的坐标。其中points为包含两个数组的元组，points[0]为对应所有的Y坐标（行号），points[1]为对应所有的X坐标（列号）
        points = np.where(mask == 255)
        if len(points[0]) == 0:
            #print(f"在mask中未找到标记点：{input_img_list[i]}")
            # 既然没有在mask中找到标注点，为什么还要在choose系列下的地址写入图片，不应该直接抛弃吗？加入噪点，增加鲁棒性
            cv2.imwrite(out_img_path, img)
            cv2.imwrite(out_mask_path, mask)
            cv2.imwrite(out_points_path, mask)
            continue

        # 0表示背景区域
        merged_result = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        correct_point = 0

        # 遍历一张图像中的每个mask中的点标注
        for i in range(len(points[0])):
            center_y, center_x = points[0][i], points[1][i]

            half_size = crop_size  
            # 切分的图像块最大只能是以点标注为中心点的20*20的区域
            x1 = max(center_x - half_size, 0)
            y1 = max(center_y - half_size, 0)
            x2 = min(center_x + half_size, img.shape[1])
            y2 = min(center_y + half_size, img.shape[0])

            roi = img[y1:y2, x1:x2]

            processed_roi, flag = process_image((center_y, center_x), (y1, y2, x1, x2), mask.shape, roi,
                                                low_threshold=20, high_threshold=40, kernel_size=(3, 3), sigma=0)

            merged_result[y1:y2, x1:x2] = merged_result[y1:y2, x1:x2] + processed_roi
            # flag为true 则表示 process_image()函数处理的图像块roi得到的掩码为有效分割
            if flag == True:
                correct_point = correct_point + 1
            else:
                continue

        # 如果用这种传统计算机视觉方法处理得到的掩码标注数量召回率>=0.8，则这张图像被认为是容易样本，
        # 并且把容易样本与其生成掩码写入train/choose/
        if (correct_point / len(points[0])) >= 0.8:
            merged_result = merged_result + mask    # 被演化的伪标签被强行与点标注图像融合
            merged_result = np.where(merged_result > 0, 255, 0)
            cv2.imwrite(out_img_path, img)
            cv2.imwrite(out_mask_path, merged_result)
            cv2.imwrite(out_points_path,mask)
        else:
            # 困难样本
            cv2.imwrite(no_choose_out_img_path, img)
            cv2.imwrite(no_choose_out_points_path,mask)

    # 容易样本数量
    print("初始数据已生成，choose(容易样本)张数：", len(os.listdir(output_image_path)))
    print("初始数据已生成，no_choose(困难样本)张数：", len(os.listdir(no_choose_output_image_path)))



# 这个函数有以下问题：
# 为什么对点标注做轮廓提取，是否多此一举？
# 另外对预测图像做轮廓提取后又填充，最后提取所有被填充的像素坐标是否也是多此一举？（做一下实验验证）
# 如果是我的话，我会用直接用cv2.connectedComponents函数来对预测图中不同连通区域打上不同标注后，与点标注图像做与"&"操作，得到的图像中的最大值就是检测成功的连通区域的个数

### copy_mask 为点提示
### target_mask为预测结果
### 比较真实标签与预测标签是否有交集，如果是，则说明识别成功。
def nc_pred_mask(copy_mask, target_mask,lose_point_ratio = 0.2,alarm_point_ration=0.2):
    # 防御：预测图与点标注尺寸不一致时，把预测图对齐到点标注尺寸
    if copy_mask.shape != target_mask.shape:
        target_mask = cv2.resize(target_mask, (copy_mask.shape[1], copy_mask.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
    copy_contours, _ = cv2.findContours(copy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target_contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 点标注没有连通域时，无法计算比例，直接判为“不可学”（避免 ZeroDivisionError）
    if len(copy_contours) == 0:
        return False

    overwrite_contours = []     # 点标注中被模型检测出的点
    un_overwrite_contours = []  # 点标注中没有被模型检测出的点

    target_index_sets = []      # 存储图像中物体的坐标
    for target_contour in target_contours:
        target_contour_mask = np.zeros(copy_mask.shape, np.uint8)
        cv2.fillPoly(target_contour_mask, [target_contour], (255))
        target_index = np.where(target_contour_mask == 255)
        target_index_XmergeY = set(target_index[0] * 1.0 + target_index[1] * 0.0001)
        target_index_sets.append(target_index_XmergeY)

    for copy_contour in copy_contours:
        copy_contour_mask = np.zeros(copy_mask.shape, np.uint8)
        cv2.fillPoly(copy_contour_mask, [copy_contour], (255))
        copy_index = np.where(copy_contour_mask == 255)
        copy_index_XmergeY = set(copy_index[0] * 1.0 + copy_index[1] * 0.0001)

        overlap_found = False
        for target_index_XmergeY in target_index_sets:
            if not copy_index_XmergeY.isdisjoint(target_index_XmergeY):     # 为True的时候，两个集合之间存在交集
                overwrite_contours.append(copy_contour)
                overlap_found = True
                break

        if not overlap_found:
            un_overwrite_contours.append(copy_contour) 

    flag = False
    if len(un_overwrite_contours) / len(copy_contours) > lose_point_ratio or ((len(target_contours) - len(overwrite_contours)) / len(copy_contours)) > alarm_point_ration:
        flag = flag         # 检查 点标注中的点被模型漏检的比例 以及 模型误报的点与实际点的比例，依据这个来作为判断模型对该样本是否检测良好的依据
    else:
        flag = True
    return flag     



def deal_pred_mask_and_true_point_in(nc_img_path,nc_pred_mask_path,nc_points_path,c_img_path,c_mask_path,c_points_path,lose_point_ratio=0.2,alarm_point_ration=0.2):
    nc_img_list = os.listdir(nc_img_path)
    new_choose_list = []
    for i in range(len(nc_img_list)):
        img_path = os.path.join(nc_img_path, nc_img_list[i])
        points_path = os.path.join(nc_points_path, nc_img_list[i])
        pred_path = os.path.join(nc_pred_mask_path, nc_img_list[i])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(points_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None or pred is None:
            print(f"图像或mask读取失败：{nc_img_list[i]}")
            continue

        # 兜底：预测图与点标注尺寸不一致时（如残留的 256×256 掩码）先对齐
        if pred.shape != mask.shape:
            pred = cv2.resize(pred, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        points = np.where(mask == 255)
        if len(points[0]) == 0:     # 如果没有点标注，就视为容易样本并进行后面的标签演化（自更新）
            new_choose_list.append(nc_img_list[i])
            continue

        flag = nc_pred_mask(mask, pred,lose_point_ratio =lose_point_ratio,alarm_point_ration=alarm_point_ration)
        if flag == True:
            new_choose_list.append(nc_img_list[i])
    return new_choose_list



###注意这边与nc_pred_mask中的相反，是因为nc_pred_mask函数要依据点标注来对预测图像做出是否检测良好的评估，而这个函数是为了
###copy_mask 为预测结果
###target_mask为点
def nc_correct_pred_mask(copy_mask, target_mask):
    # 防御：预测图与点标注尺寸不一致时，把预测图对齐到点标注尺寸
    if copy_mask.shape != target_mask.shape:
        copy_mask = cv2.resize(copy_mask, (target_mask.shape[1], target_mask.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    copy_contours, _ = cv2.findContours(copy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target_contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overwrite_contours = []     # 预测结果中的连通区域中与点标注吻合的
    un_overwrite_contours = []  # 预测结果中的连通区域中与点标注不吻合的

    target_index_sets = []
    for target_contour in target_contours:
        target_contour_mask = np.zeros(copy_mask.shape, np.uint8)
        cv2.fillPoly(target_contour_mask, [target_contour], (255))
        target_index = np.where(target_contour_mask == 255)
        target_index_XmergeY = set(target_index[0] * 1.0 + target_index[1] * 0.0001)
        target_index_sets.append(target_index_XmergeY)

    for copy_contour in copy_contours:
        copy_contour_mask = np.zeros(copy_mask.shape, np.uint8)
        cv2.fillPoly(copy_contour_mask, [copy_contour], (255))
        copy_index = np.where(copy_contour_mask == 255)
        copy_index_XmergeY = set(copy_index[0] * 1.0 + copy_index[1] * 0.0001)

        overlap_found = False
        for target_index_XmergeY in target_index_sets:
            if not copy_index_XmergeY.isdisjoint(target_index_XmergeY):
                overwrite_contours.append(copy_contour)
                overlap_found = True
                break

        if not overlap_found:
            un_overwrite_contours.append(copy_contour)

    copy_contour_mask_out = np.zeros(copy_mask.shape, np.uint8)
    for i in range(len(overwrite_contours)):
        cv2.fillPoly(copy_contour_mask_out, [overwrite_contours[i]], (255))

    copy_contour_mask_out = copy_contour_mask_out + target_mask     # 被演化的伪标签被强行与点标注图像融合
    copy_contour_mask_out = np.where(copy_contour_mask_out > 0, 255, 0)

    return copy_contour_mask_out



def deal_gen_mask_error_aera(nc_pred_mask_path,nc_points_path,new_choose_list):
    for i in range(len(new_choose_list)):
        pred_mask_path = os.path.join(nc_pred_mask_path,new_choose_list[i])
        points_path = os.path.join(nc_points_path,new_choose_list[i])

        pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
        points = cv2.imread(points_path, cv2.IMREAD_GRAYSCALE)

        if pred_mask is None or points is None:
            print(f"图像或mask读取失败：{new_choose_list[i]}")
            continue

        # 兜底：预测图与点标注尺寸不一致时先对齐，避免相加时广播报错
        if pred_mask.shape != points.shape:
            pred_mask = cv2.resize(pred_mask, (points.shape[1], points.shape[0]), interpolation=cv2.INTER_NEAREST)

        pred_mask_out = nc_correct_pred_mask(pred_mask,points)
        cv2.imwrite(pred_mask_path,pred_mask_out)
    print("生成的标签精细化完成！！！！！！")






def move_files(file_list, src_path, dst_path):
    for file_name in file_list:
        #print(file_name)
        src_file = os.path.join(src_path, file_name)
        #print(src_file)
        dst_file = os.path.join(dst_path, file_name)
        if os.path.exists(src_file):
            try:
                shutil.copy(src_file, dst_file)
                os.remove(src_file)
            except PermissionError as e:
                print(f"PermissionError: {e}")

            # shutil.move(src_file, dst_file)
        else:
            print(f"File {src_file} does not exist.")




def hard_sample_in(nc_img_path,nc_pred_mask_path,nc_points_path,c_img_path,c_mask_path,c_points_path,new_choose_list):
    move_files(new_choose_list, nc_img_path, c_img_path)
    move_files(new_choose_list, nc_pred_mask_path, c_mask_path)
    move_files(new_choose_list, nc_points_path, c_points_path)




# 注意：pred是模型得到的未经过sigmoid这种激活层的结果（新"掩码"），np.max(pred) = 0.9019772，而gt_mask是上一个自更新后的二值掩码(旧掩码)。
def update_gt_update_degen_corr(pred, gt_masks, thresh_Tb, thresh_k, size,degen=0.9):
    assert gt_masks.shape == pred.shape
    update_gt_masks = gt_masks.copy()   

    background_length = 33
    target_length = 3

    # connectedComponents函数输入参数必须是二值掩码。label_image与原图同样大小，每个连通区域里面的像素值=标签编号。num_labels表示连通区域的个数(包括了背景区域)
    num_labels, label_image = cv2.connectedComponents((gt_masks > 0.5).astype(np.uint8))

    background_kernel = np.ones((background_length, background_length), np.uint8)
    target_kernel = np.ones((target_length, target_length), np.uint8)

    pred_max = pred.max()
    max_limitation = size[0] * size[1] * 0.0015     # h*w*r

    combined_thresh_mask = np.zeros_like(pred, dtype=np.float32)
    # 该for循环的作用对于每一个旧掩码的每个连通区域，去测试对应新掩码的对应区域
    for region_num in range(1, num_labels):
        region_coords = np.argwhere(label_image == region_num)  # 找到连通区域所有的像素点坐标
        centroid = np.mean(region_coords, axis=0).astype(int)

        cur_point_mask = np.zeros_like(pred, dtype=np.uint8)
        cur_point_mask[centroid[0], centroid[1]] = 1    # 根据上一个自更新的伪标签的每个连通区域来生成点标注

        # cv2.dilate函数：用一个被称为kernel的小窗口在图像上滑动，如果窗口覆盖的区域内至少有一个像素是前景（值为1），那么窗口中心位置的像素就被设为前景
        nbr_mask = cv2.dilate(cur_point_mask, background_kernel) > 0    # nbr_mask掩码用来只关注background_kernel范围下的区域
        targets_mask = cv2.dilate(cur_point_mask, target_kernel) > 0    # targets_mask掩码用来只关注target_kernel范围下的区域

        region_size_ratio = len(region_coords) / max_limitation
        threshold_start = (pred * nbr_mask).max() * thresh_Tb
        threshold_delta = thresh_k * ((pred * nbr_mask).max() - threshold_start) * region_size_ratio
        threshold = threshold_start + threshold_delta
        threshold = threshold.cpu().numpy() if isinstance(threshold, torch.Tensor) else threshold

        thresh_mask = (pred * nbr_mask > threshold).astype(np.float32)  #这里语句的功能相当于一次对pred的自定义阈值的激活层，其中阈值与(pred * nbr_mask).max()相关

        num_labels_thresh, label_image_thresh = cv2.connectedComponents(thresh_mask.astype(np.uint8))
        
        for num_cur in range(1, num_labels_thresh):
            curr_mask = (label_image_thresh == num_cur).astype(np.float32)
            if np.sum(curr_mask * targets_mask) == 0:
                thresh_mask -= curr_mask    # 错误区域清除，检查对pred提取出的每个连通区域是否与质心(半径为3的圆)存在交集，没有就清除

        combined_thresh_mask = np.maximum(combined_thresh_mask, thresh_mask)    #保证不出现负数情况下，去融合新掩码的合适的候选区域 
    
    # combined_thresh_mask理解为根据旧掩码去更正新掩码，以下的操作根据"更正过的新掩码"来融合旧掩码，新掩码。
    target_patch = (update_gt_masks * combined_thresh_mask + pred * combined_thresh_mask) / 2
    background_patch = update_gt_masks * (1 - combined_thresh_mask)* degen
    update_gt_masks = background_patch + target_patch

    update_gt_masks = np.maximum(update_gt_masks, (gt_masks == 1).astype(np.float32))

    return update_gt_masks









