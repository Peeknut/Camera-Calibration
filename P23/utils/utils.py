# ------------------------------------------------------------------------------
# Copyright (c) NKU
# Licensed under the MIT License.
# Written by Xuanyi Li (xuanyili.edu@gmail.com)
# ------------------------------------------------------------------------------
import os
import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
def GERF_loss(GT, pred, args):
    '''
    函数作用：计算预测值与label的平均误差
    参数：
        GT：label
        pred：模型预测值

    '''
    # mask = (GT < args.maxdisp) & (GT >= 0)
    mask = GT > 0  # 01矩阵
    mask.detach_()
    count = len(torch.nonzero(mask))
    if count == 0:
        count = 1
    return torch.sum(torch.sqrt(torch.pow(GT[mask] - pred[mask], 2) + 4) /2 - 1) / count

def smooth_L1_loss(GT, pred, args):

    mask = GT < args.maxdisp
    mask.detach_()
    # loss = F.smooth_l1_loss(pred[mask], GT[mask], size_average=True)
    loss = (pred[mask] - GT[mask]).abs().mean()
    return loss

