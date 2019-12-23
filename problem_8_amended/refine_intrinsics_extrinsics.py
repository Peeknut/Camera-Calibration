#!usr/bin/env/ python
# _*_ coding:utf-8 _*_
 
import numpy as np
import math
from scipy import optimize as opt
 
#微调所有参数
def refine_intrinsics_extrinsics_param(A, W, real_coor, pic_coor):
    #整合参数
    P_init = compose_paramter_vector(A, W)

    #微调所有参数
    P = opt.leastsq(value,
                    P_init,
                    args=(W, real_coor, pic_coor),
                    Dfun=jacobian)[0]

    return decompose_paramter_vector(P)
 
#把所有参数整合到一个数组内
def compose_paramter_vector(A, W):
    alpha = np.array([A[0, 0], A[1, 1], A[0, 1], A[0, 2], A[1, 2]])
    P = alpha
    for i in range(len(W)):
        R, t = (W[i])[:, :3], (W[i])[:, 3]
        w = np.append(R.reshape((1,-1)),t)
        P = np.append(P, w)
    return P
 
 
#分解参数集合，得到对应的内参，外参，畸变矫正系数
def decompose_paramter_vector(P):
    [alpha, beta, gamma, uc, vc] = P[0:5]
    A = np.array([[alpha, gamma, uc],
                  [0, beta, vc],
                  [0, 0, 1]])
    W = []
    M = (len(P) - 5) // 12
 
    for i in range(M):
        m = 5 + 12 * i
        t = (P[m+9:m+12]).reshape(3, -1)
        R = (P[m:m+9]).reshape(3, -1)

        #依次拼接每幅图的外参
        w = np.concatenate((R, t), axis=1)
        W.append(w)
 
    W = np.array(W)
    return A, W
 
 
# 返回从真实世界坐标映射的图像坐标
def get_single_project_coor(A, W, coor):
    single_coor = np.array([coor[0], coor[1], coor[2], 1])
    uv = np.dot(np.dot(A, W), single_coor)
    uv /= uv[-1]

    return np.array([uv[0], uv[1]])

#返回所有点的真实世界坐标映射到的图像坐标与真实图像坐标的残差
def value(P, org_W, X, Y_real):
    M = (len(P) - 5) // 12
    N = len(X[0])
    A = np.array([
        [P[0], P[2], P[3]],
        [0, P[1], P[4]],
        [0, 0, 1]
    ])
    Y = np.array([])
 
    for i in range(M):
        m = 5 + 12 * i

        #取出当前图像对应的外参
        t = (P[m+9:m+12]).reshape(3, -1)
        R = (P[m:m+9]).reshape(3, -1)
        W = np.concatenate((R, t), axis=1)

        #计算每幅图的坐标残差
        for j in range(N):
            Y = np.append(Y, get_single_project_coor(A, W, (X[i])[j]))
 
    error_Y  =  np.array(Y_real).reshape(-1) - Y
 
    return error_Y
 
 
#计算对应jacobian矩阵
def jacobian(P, WW, X, Y_real):
    M = (len(P) - 5) // 12
    N = len(X[0])
    K = len(P)
    A = np.array([
        [P[0], P[2], P[3]],
        [0, P[1], P[4]],
        [0, 0, 1]
    ])
 
    res = np.array([])
 
    for i in range(M):
        m = 5 + 12* i

        t = (P[m+9:m+12]).reshape(3, -1)
        R = (P[m:m+9]).reshape(3, -1)
        W = np.concatenate((R, t), axis=1)
 
        for j in range(N):
            res = np.append(res, get_single_project_coor(A, W, (X[i])[j]))
 
    #求得x, y方向对P[k]的偏导
    J = np.zeros((K, 2 * M * N))
    for k in range(K):
        J[k] = np.gradient(res, P[k])
 
    return J.T