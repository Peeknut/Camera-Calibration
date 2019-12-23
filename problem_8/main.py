#!usr/bin/env/ python
# _*_ coding:utf-8 _*_
 
import cv2 as cv
import glob
import numpy as np
import os
from homography import get_homography
from intrinsics import get_intrinsics_param
from extrinsics import get_extrinsics_param
from distortion import get_distortion
from refine_all import refinall_all_param
 
def calibrate():
    #求单应矩阵
    H = get_homography(pic_points, real_points_x_y)
 
    #求内参
    intrinsics_param = get_intrinsics_param(H)
 
    #求对应每幅图外参
    extrinsics_param = get_extrinsics_param(H, intrinsics_param)
 
    #畸变矫正
    k = get_distortion(intrinsics_param, extrinsics_param, pic_points, real_points_x_y)
 
    #微调所有参数
    [new_intrinsics_param, new_k, new_extrinsics_param]  = refinall_all_param(intrinsics_param,
                                                            k, extrinsics_param, real_points, pic_points)
    print(len(real_points))
    print("intrinsics_parm:\t", new_intrinsics_param)
    print("distortionk:\t", new_k)
    print("extrinsics_parm:\t", new_extrinsics_param)
    print("-------------------计算反向投影误差-----------------------") 
    tot_error = 0  
    dist = np.append(new_k, [0, 0])
    for i in range(len(real_points)):  
        img_points2, _ = cv.projectPoints(real_points[i],new_extrinsics_param[i][:,:3],new_extrinsics_param[i][:,3],new_intrinsics_param,dist)  
        error = cv.norm(pic_points[i].reshape(-1,1,2),img_points2, cv.NORM_L2)/len(img_points2)  
        tot_error += error  
      
    mean_error = tot_error/len(real_points)  
    print("total error: ", tot_error)  
    print("mean error: ", mean_error )

 
if __name__ == "__main__":
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001 
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((6*7,3), np.float32)  # 7x8的格子 此处参数根据使用棋盘格规格进行修改
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2) 
    # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y  

    real_points_x_y = [] # 存储3D点x, y坐标
    real_points = [] # 存储3D点
    pic_points = [] # 存储2D点

    images = glob.glob('..\\left\\*.jpg') # 文件存储路径，存储需要标定的摄像头拍摄的棋盘格图片

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        # 寻找棋盘格角点
        ret, corners = cv.findChessboardCorners(gray, (7,6),None)

        if ret == True:
            real_points.append(objp)
            real_points_x_y.append(objp[:, :2])

            # 在原角点的基础上寻找亚像素角点
            corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            if corners2.any():
                corners2 = corners2.reshape(-1, 2)
                pic_points.append(corners2)
            else:
                corners2 = corners2.reshape(-1, 2)
                pic_points.append(corners)
    calibrate() 
