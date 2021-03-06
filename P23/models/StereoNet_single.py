# ------------------------------------------------------------------------------
# Copyright (c) NKU
# Licensed under the MIT License.
# Written by Xuanyi Li (xuanyili.edu@gmail.com)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import torch.backends.cudnn as cudnn

def convbn(in_channel, out_channel, kernel_size, stride, pad, dilation):
    
    return nn.Sequential(
        nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation>1 else pad,
            dilation=dilation),
       nn.BatchNorm2d(out_channel))

def convbn_3d(in_channel, out_channel, kernel_size, stride, pad):

    return nn.Sequential(
        nn.Conv3d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            padding=pad,
            stride=stride),
       nn.BatchNorm3d(out_channel))

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride, downsample, pad, dilation):
        super().__init__()
        self.conv1 = nn.Sequential(
            convbn(in_channel, out_channel, 3, stride, pad, dilation),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv2 = convbn(out_channel, out_channel, 3, 1, pad, dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)

        # out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        ### bug?
        out = x + out
        return out

class FeatureExtraction(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.downsample = nn.ModuleList() # 下采样
        in_channel = 3
        out_channel = 32
        for _ in range(k):
            self.downsample.append(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=5,
                    stride=2,
                    padding=2))
            in_channel = out_channel
            out_channel = 32
        self.residual_blocks = nn.ModuleList() # 残差块
        for _ in range(6):
            self.residual_blocks.append(
                BasicBlock(
                    32, 32, stride=1, downsample=None, pad=1, dilation=1))
        self.conv_alone = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
    def forward(self, rgb_img):
        output = rgb_img
        for i in range(self.k):
            output = self.downsample[i](output)
        for block in self.residual_blocks:
            output = block(output)
        return self.conv_alone(output)

class EdgeAwareRefinement(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv2d_feature = nn.Sequential(
            convbn(in_channel, 32, kernel_size=3, stride=1, pad=1, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.residual_astrous_blocks = nn.ModuleList()
        astrous_list = [1, 2, 4, 8 , 1 , 1]
        for di in astrous_list:
            self.residual_astrous_blocks.append(
                BasicBlock(
                    32, 32, stride=1, downsample=None, pad=1, dilation=di))
                
        self.conv2d_out = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, low_disparity, corresponding_rgb):
        '''
        参数：
            low_disparity：前面部分输出的结果
            corresponding_rgb：输入的左图
        '''
        output = torch.unsqueeze(low_disparity, dim=1)
        # F.interpolate根据给定 size 或 scale_factor，上采样或下采样输入数据input
        # 将 1/8 分辨率的深度图直接通过双线性插值的上采样策略上采样放大到全分辨率
        twice_disparity = F.interpolate(
            output,
            size = corresponding_rgb.size()[-2:], # 感觉最后2维是H*W
            mode='bilinear', # 双线性插值
            align_corners=False)
        if corresponding_rgb.size()[-1]/ low_disparity.size()[-1] >= 1.5:
            twice_disparity *= 8  
        output = self.conv2d_feature(
            torch.cat([twice_disparity, corresponding_rgb], dim=1))
        # 通过6个残差块
        for astrous_block in self.residual_astrous_blocks:
            output = astrous_block(output)
        # 最后经过一个 3*3 的卷积，得到最后的单通道深度图
        return nn.ReLU(inplace=True)(torch.squeeze(
            twice_disparity + self.conv2d_out(output), dim=1))

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super().__init__()
        self.disp = torch.FloatTensor(
            np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1]))
    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1)
        return out


class StereoNet(nn.Module):
    def __init__(self, k, r, maxdisp=192):
        '''
        参数：
            k: the stage num of refinement - 1
            r: the stage num of refinement - 1
        '''
        super().__init__()
        self.maxdisp = maxdisp
        self.k = k
        self.r = r
        # 网络结构1：Downsampling Feature Network k个下采样层
        self.feature_extraction = FeatureExtraction(k)
        # 网络结构2：对cost进行操作
        self.filter = nn.ModuleList()
        for _ in range(4):
            self.filter.append(
                nn.Sequential(
                    convbn_3d(32, 32, kernel_size=3, stride=1, pad=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        self.conv3d_alone = nn.Conv3d(
            32, 1, kernel_size=3, stride=1, padding=1)
        
        # 网络结构3：分层细化：边缘感知上采样
        self.edge_aware_refinements = nn.ModuleList()
        for _ in range(1):
            self.edge_aware_refinements.append(EdgeAwareRefinement(4))
    
    def forward(self, left, right):
        disp = (self.maxdisp + 1) // pow(2, self.k)
        refimg_feature = self.feature_extraction(left)
        targetimg_feature = self.feature_extraction(right)

        # matching
        cost = torch.FloatTensor(refimg_feature.size()[0],
                                 refimg_feature.size()[1],
                                 disp,
                                 refimg_feature.size()[2],
                                 refimg_feature.size()[3]).zero_()
        for i in range(disp):
            if i > 0:
                cost[:, :, i, :, i:] = refimg_feature[ :, :, :, i:] - targetimg_feature[:, :, :, :-i]
            else:
                cost[:, :, i, :, :] = refimg_feature - targetimg_feature
        cost = cost.contiguous()




        for f in self.filter:
            cost = f(cost)
        cost = self.conv3d_alone(cost)
        cost = torch.squeeze(cost, 1)
        pred = F.softmax(cost, dim=1)
        pred = disparityregression(disp)(pred)

        img_pyramid_list = [left]
        
        pred_pyramid_list= [pred]



        pred_pyramid_list.append(self.edge_aware_refinements[0](
                    pred_pyramid_list[0], img_pyramid_list[0]))

        for i in range(1):
            pred_pyramid_list[i] = pred_pyramid_list[i]* (
                left.size()[-1] / pred_pyramid_list[i].size()[-1])
            pred_pyramid_list[i] = torch.squeeze(
            F.interpolate(
                torch.unsqueeze(pred_pyramid_list[i], dim=1),
                size=left.size()[-2:],
                mode='bilinear', # 双线性插值
                align_corners=False),
            dim=1)

        return pred_pyramid_list
# if __name__ == '__main__':
#     model = StereoNet(k=3, r=4).cuda()
#     # model.eval()
#     import time
#     import datetime
#     import torch
#     # torch.backends.cudnn.benchmark = True    
#     input = torch.FloatTensor(1,3,540,960).zero_().cuda()
#     # input = torch.FloatTensor(1,3,960,512).zero_().cuda()
#     for i in range(100):
#     # pass
#         out = model(input, input)
#         # print(len(out))
#     start = datetime.datetime.now()
#     for i in range(100):
#         # pass
#         out = model(input, input)
#         # shape = [x.size() for x in out]
#         # print(shape)
#     end = datetime.datetime.now()
#     print((end-start).total_seconds())





    




