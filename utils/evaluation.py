# this file is utilized to evaluate the models from different mode: 2D-slice level, 2D-patient level, 3D-patient level
from tkinter import image_names
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import torch
import torch.nn.functional as F
import utils.metrics as metrics
from hausdorff import hausdorff_distance
from medpy.metric.binary import hd as medpy_hd
from medpy.metric.binary import hd95 as medpy_hd95
from medpy.metric.binary import assd as medpy_assd
from utils.tools import hausdorff_distance as our_hausdorff_distance
from utils.visualization import visual_segmentation, visual_segmentation_npy, visual_segmentation_binary, visual_segmentation_sets, visual_segmentation_sets_with_pt
from einops import rearrange
from utils.generate_prompts import get_click_prompt
from utils.compute_ef import compute_left_ventricle_volumes
import time
import pandas as pd
from utils.tools import corr, bias, std
from scipy import stats


# ==================== 新增开始：额外依赖 ====================
import cv2
# ==================== 新增结束 ====================


# ==================== 新增开始：时序一致性辅助函数 ====================
def temporal_dice_score(mask1, mask2, eps=1e-5):
    """
    计算两个二值掩膜之间的Dice。

    为避免连续两帧均为空掩膜时获得虚假的高时序一致性，
    此处将两个掩膜均为空的情况定义为0。
    """
    mask1 = np.asarray(mask1).astype(bool)
    mask2 = np.asarray(mask2).astype(bool)

    mask1_sum = mask1.sum()
    mask2_sum = mask2.sum()

    if mask1_sum == 0 and mask2_sum == 0:
        return 0.0

    intersection = np.logical_and(mask1, mask2).sum()

    return (
        2.0 * intersection + eps
    ) / (
        mask1_sum + mask2_sum + eps
    )


def convert_frame_to_uint8(frame):
    """
    将输入视频帧转换为OpenCV光流计算需要的灰度uint8图像。

    支持输入：
        (C, H, W)
        (H, W)
        像素范围[-1, 1]
        像素范围[0, 1]
        像素范围[0, 255]
    """
    frame = np.asarray(frame)

    if frame.ndim == 3:
        if frame.shape[0] == 1:
            frame = frame[0]
        else:
            frame = np.mean(frame, axis=0)

    frame = frame.astype(np.float32)

    frame_min = frame.min()
    frame_max = frame.max()

    if frame_min >= -1.0 and frame_max <= 1.0:
        if frame_min < 0:
            frame = (frame + 1.0) * 127.5
        else:
            frame = frame * 255.0

    frame = np.clip(frame, 0, 255).astype(np.uint8)

    return frame


def calculate_video_temporal_metrics(video, pred_masks):
    """
    计算单个视频的时序一致性指标。

    Parameters
    ----------
    video : np.ndarray
        视频图像，形状为(T, C, H, W)。

    pred_masks : np.ndarray
        完整视频的二值预测掩膜，形状为(T, H, W)。

    Returns
    -------
    tc_dice : float
        运动补偿时序一致性Dice，越高越好。

    area_smoothness : float
        归一化LV面积平滑误差，越低越好。
    """
    video = np.asarray(video)
    pred_masks = np.asarray(pred_masks).astype(np.uint8)

    common_frames = min(
        video.shape[0],
        pred_masks.shape[0]
    )

    video = video[:common_frames]
    pred_masks = pred_masks[:common_frames]

    if common_frames < 2:
        return np.nan, np.nan

    height, width = pred_masks.shape[-2:]

    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32)
    )

    temporal_dices = []

    for frame_idx in range(common_frames - 1):
        frame_current = convert_frame_to_uint8(
            video[frame_idx]
        )
        frame_next = convert_frame_to_uint8(
            video[frame_idx + 1]
        )

        # 保证视频帧和预测掩膜具有相同尺寸
        if frame_current.shape != (height, width):
            frame_current = cv2.resize(
                frame_current,
                (width, height),
                interpolation=cv2.INTER_LINEAR
            )

        if frame_next.shape != (height, width):
            frame_next = cv2.resize(
                frame_next,
                (width, height),
                interpolation=cv2.INTER_LINEAR
            )

        # 计算从下一帧到当前帧的反向光流。
        # cv2.remap根据目标像素寻找源图像中的对应位置，
        # 因此这里使用从t+1到t的反向光流。
        backward_flow = cv2.calcOpticalFlowFarneback(
            frame_next,
            frame_current,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        map_x = grid_x + backward_flow[..., 0]
        map_y = grid_y + backward_flow[..., 1]

        # 将第t帧预测掩膜变形至第t+1帧
        warped_current_mask = cv2.remap(
            pred_masks[frame_idx].astype(np.float32),
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        warped_current_mask = warped_current_mask > 0.5
        next_mask = pred_masks[frame_idx + 1] > 0

        temporal_dice = temporal_dice_score(
            warped_current_mask,
            next_mask
        )

        temporal_dices.append(temporal_dice)

    if len(temporal_dices) > 0:
        tc_dice = np.mean(temporal_dices)
    else:
        tc_dice = np.nan

    # 计算每一帧预测LV区域的面积
    areas = pred_masks.reshape(
        common_frames, -1
    ).sum(axis=1).astype(np.float64)

    mean_area = np.mean(areas)

    # 使用面积序列的归一化二阶差分评价平滑度
    if common_frames >= 3 and mean_area > 0:
        second_difference = np.diff(areas, n=2)

        area_smoothness = (
            np.mean(np.abs(second_difference)) /
            (mean_area + 1e-8)
        )
    else:
        area_smoothness = np.nan

    return tc_dice, area_smoothness
# ==================== 新增结束：时序一致性辅助函数 ====================


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def obtain_patien_id(filename):
    if "-" in filename: # filename = "xx-xx-xx_xxx"
        filename = filename.split('-')[-1]
    # filename = xxxxxxx or filename = xx_xxx
    if "_" in filename:
        patientid = filename.split("_")[0]
    else:
        patientid = filename[:3]
    return patientid

def eval_mask_slice(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    hds = np.zeros(opt.classes)
    ious, accs, ses, sps = np.zeros(opt.classes), np.zeros(opt.classes), np.zeros(opt.classes), np.zeros(opt.classes)
    eval_number = 0
    sum_time = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['low_mask'].to(dtype = torch.float32, device=opt.device))
        label = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))

        pt = get_click_prompt(datapack, opt)

        with torch.no_grad():
            start_time = time.time()
            pred = model(imgs, pt)
            sum_time =  sum_time + (time.time()-start_time)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict = torch.sigmoid(pred['masks'])
        predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = predict[:, 0, :, :] > 0.5  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            dice_i = metrics.dice_coefficient(pred_i, gt_i)
            #print("name:", name[j], "coord:", coords_torch[j], "dice:", dice_i)
            dices[1] += dice_i
            iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
            ious[1] += iou
            accs[1] += acc
            ses[1] += se
            sps[1] += sp
            hds[1] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            del pred_i, gt_i
        eval_number = eval_number + b
    dices = dices / eval_number
    hds = hds / eval_number
    ious, accs, ses, sps = ious/eval_number, accs/eval_number, ses/eval_number, sps/eval_number
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:])
    mean_hdis = np.mean(hds[1:])
    mean_iou, mean_acc, mean_se, mean_sp = np.mean(ious[1:]), np.mean(accs[1:]), np.mean(ses[1:]), np.mean(sps[1:])
    print("test speed", eval_number/sum_time)
    if opt.mode == "train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        return mean_dice, mean_iou, mean_acc, mean_se, mean_sp


def eval_mask_slice2(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    max_slice_number = opt.batch_size * (len(valloader) + 1)
    dices = np.zeros((max_slice_number, opt.classes))
    hds = np.zeros((max_slice_number, opt.classes))
    ious, accs, ses, sps = np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes))
    eval_number = 0
    sum_time = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['low_mask'].to(dtype = torch.float32, device=opt.device))
        label = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
        class_id = datapack['class_id']
        image_filename = datapack['image_name']

        pt = get_click_prompt(datapack, opt)

        with torch.no_grad():
            start_time = time.time()
            pred = model(imgs, pt)
            sum_time =  sum_time + (time.time()-start_time)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict = torch.sigmoid(pred['masks'])
        predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = predict[:, 0, :, :] > 0.5  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            dice_i = metrics.dice_coefficient(pred_i, gt_i)
            #print("name:", name[j], "coord:", coords_torch[j], "dice:", dice_i)
            dices[eval_number+j, 1] += dice_i
            iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
            ious[eval_number+j, 1] += iou
            accs[eval_number+j, 1] += acc
            ses[eval_number+j, 1] += se
            sps[eval_number+j, 1] += sp
            hds[eval_number+j, 1] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            del pred_i, gt_i
            if opt.visual:
                visual_segmentation_sets_with_pt(seg[j:j+1, :, :], image_filename[j], opt, pt[0][j, :, :])
        eval_number = eval_number + b
    dices = dices[:eval_number, :]
    hds = hds[:eval_number, :]
    ious, accs, ses, sps = ious[:eval_number, :], accs[:eval_number, :], ses[:eval_number, :], sps[:eval_number, :]
    val_losses = val_losses / (batch_idx + 1)

    dice_mean = np.mean(dices, axis=0)
    dices_std = np.std(dices, axis=0)
    hd_mean = np.mean(hds, axis=0)
    hd_std = np.std(hds, axis=0)

    mean_dice = np.mean(dice_mean[1:])
    mean_hdis = np.mean(hd_mean[1:])
    print("test speed", eval_number/sum_time)
    if opt.mode == "train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        # data = pd.DataFrame(dices*100)
        # writer = pd.ExcelWriter('./result/' + args.task + '/PT10-' + opt.modelname + '.xlsx')
        # data.to_excel(writer, 'page_1', float_format='%.2f')
        # writer._save()

        dice_mean = np.mean(dices*100, axis=0)
        dices_std = np.std(dices*100, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        iou_mean = np.mean(ious*100, axis=0)
        iou_std = np.std(ious*100, axis=0)
        acc_mean = np.mean(accs*100, axis=0)
        acc_std = np.std(accs*100, axis=0)
        se_mean = np.mean(ses*100, axis=0)
        se_std = np.std(ses*100, axis=0)
        sp_mean = np.mean(sps*100, axis=0)
        sp_std = np.std(sps*100, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std

def eval_camus_patient(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    patientnumber = 6000  # maxnum patient number
    flag = np.zeros(patientnumber)  # record the patients
    tps, fps = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    tns, fns = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    hds = np.zeros((patientnumber, opt.classes))
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['low_mask'].to(dtype = torch.float32, device=opt.device))
        label = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
        image_filename = datapack['image_name']
        class_id = datapack['class_id']

        pt = get_click_prompt(datapack, opt)
        bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)

        with torch.no_grad():
            pred = model(imgs, pt, bbox)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]

        # predict = torch.sigmoid(pred['masks'])
        # predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        # seg = predict[:, 0, :, :] > 0.5  # (b, h, w)

        predict = F.softmax(pred['masks'], dim=1)
        pred = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)

        b, h, w = seg.shape
        for j in range(0, b):
            patient_number = int(image_filename[j][:4]) # xxxx_2CH_xxx
            antrum = int(image_filename[j][5])
            if antrum == 2:
                patientid = patient_number
            elif antrum == 3:
                patientid = 2000 + patient_number
            else:
                patientid = 4000 + patient_number
            flag[patientid] = flag[patientid] + 1
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
            hd = hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            hds[patientid, class_id[j]] += hd
            tps[patientid, class_id[j]] += tp
            fps[patientid, class_id[j]] += fp
            tns[patientid, class_id[j]] += tn
            fns[patientid, class_id[j]] += fn
            if opt.visual:
                visual_segmentation(seg[j:j+1, :, :], image_filename[j], opt)
    tps = tps[flag > 0, :]
    fps = fps[flag > 0, :]
    tns = tns[flag > 0, :]
    fns = fns[flag > 0, :]
    hds = hds[flag > 0, :] / (flag[flag>0][:, None]/(opt.classes-1))
    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    dices = np.mean(patient_dices, axis=0)  # c
    hdis = np.mean(hds, axis=0)
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:], axis=0)
    mean_hdis = np.mean(hdis[1:], axis=0)
    #return dices, mean_dice, val_losses
    if opt.mode=="train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0)
        iou_std = np.std(iou, axis=0)
        acc = (tps + tns + smooth)/(tps + fps + fns + tns + smooth)
        acc_mean = np.mean(acc, axis=0)
        acc_std = np.std(acc, axis=0)
        se = (tps + smooth) / (tps + fns + smooth)
        se_mean = np.mean(se, axis=0)
        se_std = np.std(se, axis=0)
        sp = (tns + smooth) / (fps + tns + smooth)
        sp_mean = np.mean(sp, axis=0)
        sp_std = np.std(sp, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std


def eval_patient(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    patientnumber = 5000  # maxnum patient number
    flag = np.zeros(patientnumber)  # record the patients
    tps, fps = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    tns, fns = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    hds = np.zeros((patientnumber, opt.classes))
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['low_mask'].to(dtype = torch.float32, device=opt.device))
        label = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
        image_filename = datapack['image_name']
        class_id = datapack['class_id']

        pt = get_click_prompt(datapack, opt)
        bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)

        with torch.no_grad():
            pred = model(imgs, pt, bbox)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]

        predict = torch.sigmoid(pred['masks'])
        predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = predict[:, 0, :, :] > 0.5  # (b, h, w)


        # predict = F.softmax(pred['masks'], dim=1)
        # pred = predict.detach().cpu().numpy()  # (b, c, h, w)
        # seg = np.argmax(pred, axis=1)

        b, h, w = seg.shape
        for j in range(0, b):
            patientid = int(obtain_patien_id(image_filename[j]))
            flag[patientid] = flag[patientid] + 1
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
            hd = hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            hds[patientid, class_id[j]] += hd
            tps[patientid, class_id[j]] += tp
            fps[patientid, class_id[j]] += fp
            tns[patientid, class_id[j]] += tn
            fns[patientid, class_id[j]] += fn
            if opt.visual:
                visual_segmentation(seg[j:j+1, :, :], image_filename[j], opt)
    tps = tps[flag > 0, :]
    fps = fps[flag > 0, :]
    tns = tns[flag > 0, :]
    fns = fns[flag > 0, :]
    hds = hds[flag > 0, :] / (flag[flag>0][:, None]/(opt.classes-1))
    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    dices = np.mean(patient_dices, axis=0)  # c
    hdis = np.mean(hds, axis=0)
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:], axis=0)
    mean_hdis = np.mean(hdis[1:], axis=0)
    #return dices, mean_dice, val_losses
    if opt.mode=="train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0)
        iou_std = np.std(iou, axis=0)
        acc = (tps + tns + smooth)/(tps + fps + fns + tns + smooth)
        acc_mean = np.mean(acc, axis=0)
        acc_std = np.std(acc, axis=0)
        se = (tps + smooth) / (tps + fns + smooth)
        se_mean = np.mean(se, axis=0)
        se_std = np.std(se, axis=0)
        sp = (tns + smooth) / (fps + tns + smooth)
        sp_mean = np.mean(sp, axis=0)
        sp_std = np.std(sp, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std


def eval_slice(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    max_slice_number = opt.batch_size * (len(valloader) + 1)
    dices = np.zeros((max_slice_number, opt.classes))
    hds = np.zeros((max_slice_number, opt.classes))
    ious, accs, ses, sps = np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes))
    eval_number = 0
    sum_time = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = datapack['image'].to(dtype = torch.float32, device=opt.device)
        masks = datapack['low_mask'].to(dtype = torch.float32, device=opt.device)
        label = datapack['label'].to(dtype = torch.float32, device=opt.device)
        pt = get_click_prompt(datapack, opt)
        image_filename = datapack['image_name']

        with torch.no_grad():
            start_time = time.time()
            pred = model(imgs, pt)
            sum_time =  sum_time + (time.time()-start_time)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict_masks = pred['masks']
        predict_masks = torch.softmax(predict_masks, dim=1)
        pred = predict_masks.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            dices[eval_number+j, 1] += metrics.dice_coefficient(pred_i, gt_i)
            iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
            ious[eval_number+j, 1] += iou
            accs[eval_number+j, 1] += acc
            ses[eval_number+j, 1] += se
            sps[eval_number+j, 1] += sp
            hds[eval_number+j, 1] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            del pred_i, gt_i
            if opt.visual:
                visual_segmentation_sets_with_pt(seg[j:j+1, :, :], image_filename[j], opt, pt[0][j, :, :])
        eval_number = eval_number + b
    dices = dices[:eval_number, :]
    hds = hds[:eval_number, :]
    ious, accs, ses, sps = ious[:eval_number, :], accs[:eval_number, :], ses[:eval_number, :], sps[:eval_number, :]
    val_losses = val_losses / (batch_idx + 1)

    dice_mean = np.mean(dices, axis=0)
    dices_std = np.std(dices, axis=0)
    hd_mean = np.mean(hds, axis=0)
    hd_std = np.std(hds, axis=0)

    mean_dice = np.mean(dice_mean[1:])
    mean_hdis = np.mean(hd_mean[1:])
    print("test speed", eval_number/sum_time)
    if opt.mode == "train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        # data = pd.DataFrame(dices*100)
        # writer = pd.ExcelWriter('./result/' + args.task + '/PT10-' + opt.modelname + '.xlsx')
        # data.to_excel(writer, 'page_1', float_format='%.2f')
        # writer._save()

        dice_mean = np.mean(dices*100, axis=0)
        dices_std = np.std(dices*100, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        iou_mean = np.mean(ious*100, axis=0)
        iou_std = np.std(ious*100, axis=0)
        acc_mean = np.mean(accs*100, axis=0)
        acc_std = np.std(accs*100, axis=0)
        se_mean = np.mean(ses*100, axis=0)
        se_std = np.std(ses*100, axis=0)
        sp_mean = np.mean(sps*100, axis=0)
        sp_std = np.std(sps*100, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std


def eval_camus_samed(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    classes = 4
    dices = np.zeros(classes)
    patientnumber = 6000  # maxnum patient number
    flag = np.zeros(patientnumber)  # record the patients
    tps, fps = np.zeros((patientnumber, classes)), np.zeros((patientnumber, classes))
    tns, fns = np.zeros((patientnumber, classes)), np.zeros((patientnumber, classes))
    hds = np.zeros((patientnumber, classes))
    eval_number = 0
    sum_time = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = datapack['image'].to(dtype = torch.float32, device=opt.device)
        masks = datapack['low_mask'].to(dtype = torch.float32, device=opt.device)
        label = datapack['label'].to(dtype = torch.float32, device=opt.device)
        image_filename = datapack['image_name']
        class_id = datapack['class_id']

        pt = get_click_prompt(datapack, opt)
        bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)

        with torch.no_grad():
            start_time = time.time()
            pred = model(imgs, pt, bbox)
            sum_time =  sum_time + (time.time()-start_time)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict_masks = pred['masks']
        predict_masks = torch.softmax(predict_masks, dim=1)
        pred = predict_masks.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            patient_number = int(image_filename[j][:4]) # xxxx_2CH_xxx
            antrum = int(image_filename[j][5])
            if antrum == 2:
                patientid = patient_number
            elif antrum ==3:
                patientid = 2000 + patient_number
            else:
                patientid = 4000 + patient_number
            flag[patientid] = flag[patientid] + 1
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
            hd = hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            hds[patientid, class_id[j]] += hd
            tps[patientid, class_id[j]] += tp
            fps[patientid, class_id[j]] += fp
            tns[patientid, class_id[j]] += tn
            fns[patientid, class_id[j]] += fn
            if opt.visual:
                visual_segmentation(seg[j:j+1, :, :], image_filename[j], opt)
        eval_number = eval_number + b
    tps = tps[flag > 0, :]
    fps = fps[flag > 0, :]
    tns = tns[flag > 0, :]
    fns = fns[flag > 0, :]
    hds = hds[flag > 0, :] / (flag[flag>0][:, None]/(opt.classes-1))
    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    dices = np.mean(patient_dices, axis=0)  # c
    hdis = np.mean(hds, axis=0)
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:], axis=0)
    mean_hdis = np.mean(hdis[1:], axis=0)
    #return dices, mean_dice, val_losses
    print("test speed", eval_number/sum_time)
    if opt.mode=="train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0)
        iou_std = np.std(iou, axis=0)
        acc = (tps + tns + smooth)/(tps + fps + fns + tns + smooth)
        acc_mean = np.mean(acc, axis=0)
        acc_std = np.std(acc, axis=0)
        se = (tps + smooth) / (tps + fns + smooth)
        se_mean = np.mean(se, axis=0)
        se_std = np.std(se, axis=0)
        sp = (tns + smooth) / (fps + tns + smooth)
        sp_mean = np.mean(sp, axis=0)
        sp_std = np.std(sp, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std


def eval_camus(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    tps, fps, tns, fns, hds, assds= [],[],[],[],[],[]
    mask_dict = {}
    gt_efs = {}
    sum_time = 0.0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
        spcaing = datapack['spacing'].detach().cpu().numpy()[0,:2][::-1] # remove z and reverse (y,x)
        # video to image
        b, t, c, h, w = imgs.shape
        i=0

        image_filename = datapack['image_name']
        patient_name = image_filename[0].split('.')[0].split('_')[0]
        view = image_filename[0].split('.')[0].split('_')[1]
        # gt_efs[patient_name] = datapack['ef'].detach().cpu().numpy()[0]
        class_id = datapack['class_id']
        if args.disable_point_prompt:
            # pt[0]: b t 1 2
            # pt[1]: t 1
            pt = None
        else:
            pt = get_click_prompt(datapack, opt)

        start = time.time()
        with torch.no_grad():
            pred = model(imgs, pt, None)
        end = time.time()
        print(i,'infer_time:', (end-start))
        sum_time = sum_time + (end-start)

        # continue
        # semi
        # opt.semi = True

        masks = masks / 255.0
        val_loss = criterion(pred[:,:,0], masks)
        if opt.semi:
            pred = pred[:,[0,-1]]
            masks = masks[:,[0,-1]]
        val_losses += val_loss.item()

        gt = masks.detach().cpu().numpy()
        predict = F.sigmoid(pred[:,:,0,:,:])
        predict = predict.detach().cpu().numpy()  # (b, t, h, w)
        seg = predict > 0.6

        seg_mask = np.zeros_like(gt)
        seg_mask[seg] = 1
        if patient_name not in mask_dict:
            mask_dict[patient_name] = {}
        mask_dict[patient_name][view] = {'ED':seg_mask[0,0], 'ES':seg_mask[0,-1],'spacing':spcaing}

        b, t, h, w = seg.shape

        for j in range(0, b):
            for idx, frame_i in enumerate(range(0,t)):
                # for idx, frame_i in enumerate([0,t-1]):
                pred_i = np.zeros((1, h, w))
                pred_i[seg[j:j+1, frame_i,:, :] == 1] = 255

                gt_i = np.zeros((1, h, w))
                gt_i[gt[j:j+1, frame_i, :, :] == 1] = 255
                tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
                # hausdorff_distance
                # hd = hausdorff_distance(pred_i[0], gt_i[0], distance="euclidean")
                # our
                # our_hd = our_hausdorff_distance(pred_i[0], gt_i[0], percentile=100)
                # our_hd_95 = our_hausdorff_distance(pred_i[0], gt_i[0], percentile=95)
                # medpy
                # med_hd = medpy_hd(pred_i[0], gt_i[0], voxelspacing=spcaing)
                if not (np.any(pred_i[0]) and np.any(gt_i[0])):
                    print("Skipping this sample because it has no binary objects.")
                    continue

                if opt.mode == "test":
                    try:
                        med_hd95 = medpy_hd95(pred_i[0], gt_i[0], voxelspacing=spcaing)
                        # med_assd = medpy_assd(pred_i[0], gt_i[0], voxelspacing=spcaing)
                    except:
                        # print(pred_i[0], gt_i[0])
                        raise RuntimeError
                    hds.append(med_hd95)
                    # assds.append(med_assd)
                tps.append(tp)
                fps.append(fp)
                tns.append(tn)
                fns.append(fn)
                dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
                # print(dice)
                if opt.visual:
                    visual_segmentation_npy(pred_i[0,...], gt_i[0,...], image_filename[j], opt, imgs[j:j+1, frame_i, :, :, :], frameidx=frame_i)
    
    print('average_fps:', 1/ (sum_time / len(valloader) / 10) )
    tps = np.array(tps)
    fps = np.array(fps)
    tns = np.array(tns)
    fns = np.array(fns)
    hds = np.array(hds)
    assds = np.array(assds)
    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    #return dices, mean_dice, val_losses


    if opt.mode == "train":
        
        dices = np.mean(patient_dices, axis=0)  # c
        hdis = np.mean(hds, axis=0)
        val_losses = val_losses / (batch_idx + 1)
        mean_dice = dices[0]
        mean_hdis = hdis
        # return dices, mean_dice, val_losses
        return dices, mean_dice, mean_hdis, val_losses
    else:
        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0)
        iou_std = np.std(iou, axis=0)
        assd_mean = np.mean(assds, axis=0)
        assd_std = np.std(assds, axis=0)
        return dice_mean, iou_mean, hd_mean, assd_mean, dices_std, iou_std, hd_std, assd_std




def eval_echonet(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    tps, fps, tns, fns, hds, assds= [],[],[],[],[],[]
    mask_dict = {}
    gt_efs = {}
    sum_time = 0
    for batch_idx, (datapack) in enumerate(valloader):
        torch.cuda.empty_cache()
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
        # print(masks)
        spcaing = datapack['spacing'].detach().cpu().numpy()[0,:2][::-1] # remove z and reverse (y,x)
        # video to image
        # b, t, c, h, w = imgs.shape

        image_filename = datapack['image_name']
        image_name = image_filename[0].split(".")[0]

        # gt_efs[image_name] = datapack['ef'].detach().cpu().numpy()[0]
        # if args.enable_point_prompt:
        #     # pt[0]: b t 1 2
        #     # pt[1]: t 1
        pt = get_click_prompt(datapack, opt)
        # print("pt",pt)
        # else:
        # pt = None
        import time
        start = time.time()
        with torch.no_grad():
            pred = model(imgs, pt, None)
            
        torch.cuda.synchronize()
        end = time.time()
        sum_time = sum_time +(end-start)
        print('infer_time:', end-start)

        # masks = masks / 255.0
        if opt.semi:
            pred = pred[:,[0,-1]]
            masks = masks[:,[0,-1]]
        # else:
        #     # insert fake frame
        #     masks_zero = torch.zeros_like(pred,dtype=torch.uint8)
        #     masks_zero = masks_zero[:,:,0]
        #     masks_zero[:,0] = masks[:,0]
        #     masks_zero[:,-1] = masks[:,-1]
        #     masks = masks_zero

        val_loss = criterion(pred[:,:,0], masks)
        val_losses += val_loss.item()

        gt = masks.detach().cpu().numpy()
        predict = F.sigmoid(pred[:,:,0,:,:])
        predict = predict.detach().cpu().numpy()  # (b, t, h, w)
        seg = predict > 0.3
        # seg = predict > 0.6

        seg_mask = np.zeros_like(gt)
        seg_mask[seg] = 1
        if image_name not in mask_dict:
            mask_dict[image_name] = {}
        mask_dict[image_name] = {'ED':seg_mask[0,0], 'ES':seg_mask[0,-1],'spacing':spcaing}

        b, t, h, w = seg.shape
        flag = False
        for j in range(0, b):
            for idx, frame_i in enumerate(range(0,t)):
                # for idx, frame_i in enumerate([0,t-1]):
                pred_i = np.zeros((1, h, w))
                pred_i[seg[j:j+1, frame_i,:, :] == 1] = 255
                gt_i = np.zeros((1, h, w))
                gt_i[gt[j:j+1, frame_i, :, :] == 1] = 255
                tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
                
                dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
                # print(dice)
                if opt.visual:
                    visual_segmentation_npy(pred_i[0, ...],
                                            gt_i[0, ...],
                                            image_filename[j],
                                            opt,
                                            imgs[j:j + 1, frame_i, :, :, :],
                                            frameidx=frame_i)
                    
                if not (np.any(pred_i[0]) and np.any(gt_i[0])):
                    # print("Skipping this sample because it has no binary objects.")
                    continue

                # hausdorff_distance
                # hd = hausdorff_distance(pred_i[0], gt_i[0], distance="euclidean")
                # our
                # our_hd = our_hausdorff_distance(pred_i[0], gt_i[0], percentile=100)
                # our_hd_95 = our_hausdorff_distance(pred_i[0], gt_i[0], percentile=95)
                # medpy
                # med_hd = medpy_hd(pred_i[0], gt_i[0], voxelspacing=spcaing)
                try:
                    med_hd95 = medpy_hd95(pred_i[0], gt_i[0], voxelspacing=spcaing)
                    # med_assd = medpy_assd(pred_i[0], gt_i[0], voxelspacing=spcaing)
                except:
                    # print(pred_i[0], gt_i[0])
                    raise RuntimeError
                # print(med_hd95)
                # print(med_assd)
                hds.append(med_hd95)
                # assds.append(med_assd)
                tps.append(tp)
                fps.append(fp)
                tns.append(tn)
                fns.append(fn)
                dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
                # print(dice)

    print(sum_time / len(valloader))
    tps = np.array(tps)
    fps = np.array(fps)
    tns = np.array(tns)
    fns = np.array(fns)
    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    #return dices, mean_dice, val_losses
    if opt.mode == "train":
        dices = np.mean(patient_dices, axis=0)  # c
        hdis = np.mean(hds, axis=0)
        val_losses = val_losses / (batch_idx + 1)
        # mean_dice = dices[0]
        mean_dice = dices.item()
        mean_hdis = hdis
        return dices, mean_dice, mean_hdis, val_losses
        # return dices, mean_dice, val_losses
    else:
        # hds = np.array(hds)
        # assds = np.array(assds)
        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)

        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0)
        iou_std = np.std(iou, axis=0)
        assd_mean = np.mean(assds, axis=0)
        assd_std = np.std(assds, axis=0)

        return dice_mean, iou_mean, hd_mean, assd_mean, dices_std, iou_std, hd_std, assd_std



def eval_echonet_pediatric(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    tps, fps, tns, fns, hds, assds= [],[],[],[],[],[]
    mask_dict = {}
    gt_efs = {}
    sum_time = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
        spcaing = datapack['spacing'].detach().cpu().numpy()[0,:2][::-1] # remove z and reverse (y,x)
       
        image_filename = datapack['image_name']
        image_name = image_filename[0].split(".")[0]

        pt = get_click_prompt(datapack, opt)
    
        import time
        start = time.time()
        with torch.no_grad():
            pred = model(imgs, pt, None)
        end = time.time()
        sum_time = sum_time +(end-start)
        print('infer_time:', end-start)

        # masks = masks / 255.0
        if opt.semi:
            pred = pred[:,[0,-1]]
            masks = masks[:,[0,-1]]
        else:
            # insert fake frame
            masks_zero = torch.zeros_like(pred,dtype=torch.uint8)
            masks_zero = masks_zero[:,:,0]
            masks_zero[:,0] = masks[:,0]
            masks_zero[:,-1] = masks[:,-1]
            masks = masks_zero

        # val_loss = criterion(pred[:,:,0], masks)
        # val_losses += val_loss.item()

        gt = masks.detach().cpu().numpy()
        predict = F.sigmoid(pred[:,:,0,:,:])
        predict = predict.detach().cpu().numpy()  # (b, t, h, w)
        seg = predict > 0.6

        seg_mask = np.zeros_like(gt)
        seg_mask[seg] = 1
        # if image_name not in mask_dict:
        #     mask_dict[image_name] = {}
        # mask_dict[image_name] = {'ED':seg_mask[0,0], 'ES':seg_mask[0,-1],'spacing':spcaing}

        b, t, h, w = seg.shape
        flag = False
        for j in range(0, b):
            for idx, frame_i in enumerate(range(0,t)):
                # for idx, frame_i in enumerate([0,t-1]):
                pred_i = np.zeros((1, h, w))
                pred_i[seg[j:j+1, frame_i,:, :] == 1] = 255
                gt_i = np.zeros((1, h, w))
                gt_i[gt[j:j+1, frame_i, :, :] == 1] = 255
                tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)

                dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
                # print(dice)
                if opt.visual:
                    visual_segmentation_npy(pred_i[0, ...],
                                            gt_i[0, ...],
                                            image_filename[j],
                                            opt,
                                            imgs[j:j + 1, frame_i, :, :, :],
                                            frameidx=frame_i)
                    
                if not (np.any(pred_i[0]) and np.any(gt_i[0])):
                    # print("Skipping this sample because it has no binary objects.")
                    continue

                # hausdorff_distance
                # hd = hausdorff_distance(pred_i[0], gt_i[0], distance="euclidean")
                # our
                # our_hd = our_hausdorff_distance(pred_i[0], gt_i[0], percentile=100)
                # our_hd_95 = our_hausdorff_distance(pred_i[0], gt_i[0], percentile=95)
                # medpy
                # med_hd = medpy_hd(pred_i[0], gt_i[0], voxelspacing=spcaing)
                try:
                    med_hd95 = medpy_hd95(pred_i[0], gt_i[0], voxelspacing=spcaing)
                    # med_assd = medpy_assd(pred_i[0], gt_i[0], voxelspacing=spcaing)
                except:
                    # print(pred_i[0], gt_i[0])
                    raise RuntimeError
                # print(med_hd95)
                # print(med_assd)
                hds.append(med_hd95)
                # assds.append(med_assd)
                tps.append(tp)
                fps.append(fp)
                tns.append(tn)
                fns.append(fn)
                dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
                # print(dice)

    print(sum_time / len(valloader))
    tps = np.array(tps)
    fps = np.array(fps)
    tns = np.array(tns)
    fns = np.array(fns)
    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    #return dices, mean_dice, val_losses
    if opt.mode == "train":
        dices = np.mean(patient_dices, axis=0)  # c
        hdis = np.mean(hds, axis=0)
        val_losses = val_losses / (batch_idx + 1)
        mean_dice = dices[0]
        mean_hdis = hdis
        return dices, mean_dice, mean_hdis, val_losses
        # return dices, mean_dice, val_losses
    else:
        # hds = np.array(hds)
        # assds = np.array(assds)
        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)

        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0)
        iou_std = np.std(iou, axis=0)
        assd_mean = np.mean(assds, axis=0)
        assd_std = np.std(assds, axis=0)

        return dice_mean, iou_mean, hd_mean, assd_mean, dices_std, iou_std, hd_std, assd_std




def eval_hmcqu(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    tps, fps, tns, fns, hds, assds= [],[],[],[],[],[]
    mask_dict = {}
    # gt_efs = {}
    sum_time = 0
    count = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
       
        masks = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
        
        # spcaing = datapack['spacing'].detach().cpu().numpy()[0,:2][::-1] # remove z and reverse (y,x)
        
        # video to image
        # b, t, c, h, w = imgs.shape

        image_filename = datapack['image_name']
        image_name = image_filename[0].split(".")[0]

        pt1 = torch.tensor([[[[110., 110.]], [[106., 111.]]]], device='cuda:0')
        pt2 = torch.tensor([[1], [1]], device='cuda:0', dtype=torch.int32)

        pt = (pt1,pt2)

        import time
        start = time.time()
        with torch.no_grad():
            count = count +1
            pred = model(imgs, pt, None)
        end = time.time()
        sum_time = sum_time +(end-start)
        
        print(count,'infer_time:', end-start)

        masks = masks / 255.0
        if opt.semi:
            pred = pred[:,[0,-1]]
            masks = masks[:,[0,-1]]
            masks = masks.float()
        else:
            # insert fake frame
            masks_zero = torch.zeros_like(pred,dtype=torch.uint8)
            masks_zero = masks_zero[:,:,0]
            masks_zero[:,0] = masks[:,0]
            masks_zero[:,-1] = masks[:,-1]
            masks = masks_zero
            masks = masks.float()

        val_loss = criterion(pred[:,:,0], masks)
        val_losses += val_loss.item()
        
        gt = masks.detach().cpu().numpy()
        predict = F.sigmoid(pred[:,:,0,:,:])
        predict = predict.detach().cpu().numpy()  # (b, t, h, w)
        seg = predict > 0.6

        seg_mask = np.zeros_like(gt)
        seg_mask[seg] = 1
        if image_name not in mask_dict:
            mask_dict[image_name] = {}
        # mask_dict[image_name] = {'ED':seg_mask[0,0], 'ES':seg_mask[0,-1],'spacing':spcaing}

        b, t, h, w = seg.shape
       
        flag = False
        for j in range(0, b):
            for idx, frame_i in enumerate(range(0,t)):
                # for idx, frame_i in enumerate([0,t-1]):
                pred_i = np.zeros((1, h, w))
                pred_i[seg[j:j+1, frame_i,:, :] == 1] = 255
                gt_i = np.zeros((1, h, w))
                gt_i[gt[j:j+1, frame_i, :, :] == 1] = 255
                tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)

                dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
                # print(dice)
                if opt.visual:
                    visual_segmentation_npy(pred_i[0, ...],
                                            gt_i[0, ...],
                                            image_filename[j],
                                            opt,
                                            imgs[j:j + 1, frame_i, :, :, :],
                                            frameidx=frame_i)
            

                if not (np.any(pred_i[0]) and np.any(gt_i[0])):
                    # print(f"Skipping sample [{j},{frame_i}] because it has no binary objects.")
                    continue
                try:
                    med_hd95 = medpy_hd95(pred_i[0], gt_i[0])
                    # med_assd = medpy_assd(pred_i[0], gt_i[0])
                except:
                    # print(pred_i[0], gt_i[0])
                    raise RuntimeError
                # print(med_hd95)
                # print(med_assd)
                hds.append(med_hd95)
                # assds.append(med_assd)
                tps.append(tp)
                fps.append(fp)
                tns.append(tn)
                fns.append(fn)
                dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
                # print(dice)
               

    print(sum_time / len(valloader))
    tps = np.array(tps)
    fps = np.array(fps)
    tns = np.array(tns)
    fns = np.array(fns)

    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    #return dices, mean_dice, val_losses
    if opt.mode == "train":
        dices = np.mean(patient_dices, axis=0)  # c
        hdis = np.mean(hds, axis=0)
        val_losses = val_losses / (batch_idx + 1)
        mean_dice = dices[0]
        mean_hdis = hdis
        return dices, mean_dice, mean_hdis, val_losses
        # return dices, mean_dice, val_losses
    else:
        # hds = np.array(hds)
        # assds = np.array(assds)
        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0)
        iou_std = np.std(iou, axis=0)
        assd_mean = np.mean(assds, axis=0)
        assd_std = np.std(assds, axis=0)

        return dice_mean, iou_mean, hd_mean, assd_mean, dices_std, iou_std, hd_std, assd_std
    

def eval_fahgm_gd2h(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    tps, fps, tns, fns, hds, assds= [],[],[],[],[],[]
    mask_dict = {}
    # gt_efs = {}
    sum_time = 0
    count = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        
        masks = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
        # print(torch.all(masks == 0))
        
        # spcaing = datapack['spacing'].detach().cpu().numpy()[0,:2][::-1] # remove z and reverse (y,x)
        
        # video to image
        # b, t, c, h, w = imgs.shape

        image_filename = datapack['image_name']
        image_name = image_filename[0].split(".")[0]
        
        # pt1 = torch.tensor([[[[110., 110.]], [[106., 111.]]]], device='cuda:0')
        # pt2 = torch.tensor([[1], [1]], device='cuda:0', dtype=torch.int32)
        # pt = (pt1,pt2)
        pt = get_click_prompt(datapack, opt)

        import time
        start = time.time()
        with torch.no_grad():
            count = count +1
            pred = model(imgs, pt, None)
        end = time.time()
        sum_time = sum_time +(end-start)
        
        print(count,'infer_time:', end-start)

        # masks = masks * 255.0
        if opt.semi:
            pred = pred[:,[0,-1]]
            masks = masks[:,[0,-1]]
           
        
        # print(torch.all(masks == 0))

        val_loss = criterion(pred[:,:,0], masks)
        val_losses += val_loss.item()
        
        gt = masks.detach().cpu().numpy()
        predict = F.sigmoid(pred[:,:,0,:,:])
        predict = predict.detach().cpu().numpy()  # (b, t, h, w)
        seg = predict > 0.6

        seg_mask = np.zeros_like(gt)
        seg_mask[seg] = 1
        if image_name not in mask_dict:
            mask_dict[image_name] = {}
        # mask_dict[image_name] = {'ED':seg_mask[0,0], 'ES':seg_mask[0,-1],'spacing':spcaing}

        b, t, h, w = seg.shape
       
        flag = False
        for j in range(0, b):
            for idx, frame_i in enumerate(range(0,t)):
                # for idx, frame_i in enumerate([0,t-1]):
                pred_i = np.zeros((1, h, w))
                pred_i[seg[j:j+1, frame_i,:, :] == 1] = 255
                gt_i = np.zeros((1, h, w))
                gt_i[gt[j:j+1, frame_i, :, :] == 1] = 255
                tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)

                dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
                # print(dice)
                if opt.visual:
                    visual_segmentation_npy(pred_i[0, ...],
                                            gt_i[0, ...],
                                            image_filename[j],
                                            opt,
                                            imgs[j:j + 1, frame_i, :, :, :],
                                            frameidx=frame_i)
            

                if not (np.any(pred_i[0]) and np.any(gt_i[0])):
                    # print(f"Skipping sample [{j},{frame_i}] because it has no binary objects.")
                    continue
                try:
                    med_hd95 = medpy_hd95(pred_i[0], gt_i[0])
                    # med_assd = medpy_assd(pred_i[0], gt_i[0])
                except:
                    # print(pred_i[0], gt_i[0])
                    raise RuntimeError
                # print(med_hd95)
                # print(med_assd)
                hds.append(med_hd95)
                # assds.append(med_assd)
                tps.append(tp)
                fps.append(fp)
                tns.append(tn)
                fns.append(fn)
                dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
                # print(dice)
               

    print(sum_time / len(valloader))
    tps = np.array(tps)
    fps = np.array(fps)
    tns = np.array(tns)
    fns = np.array(fns)

    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    #return dices, mean_dice, val_losses
    if opt.mode == "train":
        dices = np.mean(patient_dices, axis=0)  # c
        hdis = np.mean(hds, axis=0)
        val_losses = val_losses / (batch_idx + 1)
        mean_dice = dices[0]
        mean_hdis = hdis
        return dices, mean_dice, mean_hdis, val_losses
        # return dices, mean_dice, val_losses
    else:
        # hds = np.array(hds)
        # assds = np.array(assds)
        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0)
        iou_std = np.std(iou, axis=0)
        assd_mean = np.mean(assds, axis=0)
        assd_std = np.std(assds, axis=0)

        return dice_mean, iou_mean, hd_mean, assd_mean, dices_std, iou_std, hd_std, assd_std




def eval_hmcqu_gd2h(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    tps, fps, tns, fns, hds, assds= [],[],[],[],[],[]
    mask_dict = {}
    # gt_efs = {}
    sum_time = 0
    count = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        
        masks = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
        # print(torch.all(masks == 0))
        
        # spcaing = datapack['spacing'].detach().cpu().numpy()[0,:2][::-1] # remove z and reverse (y,x)
        
        # video to image
        # b, t, c, h, w = imgs.shape

        image_filename = datapack['image_name']
        image_name = image_filename[0].split(".")[0]

        pt1 = torch.tensor([[[[110., 110.]], [[106., 111.]]]], device='cuda:0')
        pt2 = torch.tensor([[1], [1]], device='cuda:0', dtype=torch.int32)
        pt = (pt1,pt2)
        # pt = get_click_prompt(datapack, opt)

        import time
        start = time.time()
        with torch.no_grad():
            count = count +1
            pred = model(imgs, pt, None)
        end = time.time()
        sum_time = sum_time +(end-start)
        
        print(count,'infer_time:', end-start)

        # masks = masks * 255.0
        if opt.semi:
            pred = pred[:,[0,-1]]
            masks = masks[:,[0,-1]]
           
        
        # print(torch.all(masks == 0))

        val_loss = criterion(pred[:,:,0], masks)
        val_losses += val_loss.item()
        
        gt = masks.detach().cpu().numpy()
        predict = F.sigmoid(pred[:,:,0,:,:])
        predict = predict.detach().cpu().numpy()  # (b, t, h, w)
        seg = predict > 0.6

        seg_mask = np.zeros_like(gt)
        seg_mask[seg] = 1
        if image_name not in mask_dict:
            mask_dict[image_name] = {}
        # mask_dict[image_name] = {'ED':seg_mask[0,0], 'ES':seg_mask[0,-1],'spacing':spcaing}

        b, t, h, w = seg.shape
       
        flag = False
        for j in range(0, b):
            for idx, frame_i in enumerate(range(0,t)):
                # for idx, frame_i in enumerate([0,t-1]):
                pred_i = np.zeros((1, h, w))
                pred_i[seg[j:j+1, frame_i,:, :] == 1] = 255
                gt_i = np.zeros((1, h, w))
                gt_i[gt[j:j+1, frame_i, :, :] == 1] = 255
                tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)

                dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
                # print(dice)
                if opt.visual:
                    visual_segmentation_npy(pred_i[0, ...],
                                            gt_i[0, ...],
                                            image_filename[j],
                                            opt,
                                            imgs[j:j + 1, frame_i, :, :, :],
                                            frameidx=frame_i)
            

                if not (np.any(pred_i[0]) and np.any(gt_i[0])):
                    # print(f"Skipping sample [{j},{frame_i}] because it has no binary objects.")
                    continue
                try:
                    med_hd95 = medpy_hd95(pred_i[0], gt_i[0])
                    # med_assd = medpy_assd(pred_i[0], gt_i[0])
                except:
                    # print(pred_i[0], gt_i[0])
                    raise RuntimeError
                # print(med_hd95)
                # print(med_assd)
                hds.append(med_hd95)
                # assds.append(med_assd)
                tps.append(tp)
                fps.append(fp)
                tns.append(tn)
                fns.append(fn)
                dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
                # print(dice)
               

    print(sum_time / len(valloader))
    tps = np.array(tps)
    fps = np.array(fps)
    tns = np.array(tns)
    fns = np.array(fns)

    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    #return dices, mean_dice, val_losses
    if opt.mode == "train":
        dices = np.mean(patient_dices, axis=0)  # c
        hdis = np.mean(hds, axis=0)
        val_losses = val_losses / (batch_idx + 1)
        mean_dice = dices[0]
        mean_hdis = hdis
        return dices, mean_dice, mean_hdis, val_losses
        # return dices, mean_dice, val_losses
    else:
        # hds = np.array(hds)
        # assds = np.array(assds)
        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0)
        iou_std = np.std(iou, axis=0)
        assd_mean = np.mean(assds, axis=0)
        assd_std = np.std(assds, axis=0)

        return dice_mean, iou_mean, hd_mean, assd_mean, dices_std, iou_std, hd_std, assd_std


def get_eval(valloader, model, criterion, opt, args):
    if args.modelname == "SAMed":
        if opt.eval_mode == "camusmulti":
            opt.eval_mode = "camus_samed"
        else:
            opt.eval_mode = "slice"
    if opt.eval_mode == "mask_slice":
        return eval_mask_slice2(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "slice":
        return eval_slice(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "camusmulti":
        return eval_camus_patient(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "patient":
        return eval_patient(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "camus_samed":
        return eval_camus_samed(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "echonet":
        return eval_echonet(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "camus":
        return eval_camus(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "hmcqu":
        return eval_hmcqu(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "echonet_pediatric":
        return eval_echonet_pediatric(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "hmcqu_gd2h":
        return eval_hmcqu_gd2h(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "fahgm_gd2h":
        return eval_fahgm_gd2h(valloader, model, criterion, opt, args)
    else:
        raise RuntimeError("Could not find the eval mode:", opt.eval_mode)