#!usr/bin/env/ python
# _*_ coding:utf-8 _*_
 
import numpy as np
import math
from scipy import optimize as opt
 
#微调所有参数
def refinall_all_param(A, k, W, real_coor, pic_coor):
    #整合参数
    P_init = compose_paramter_vector(A, k, W)
 
    #微调所有参数
    P = opt.leastsq(value,
                    P_init,
                    args=(W, real_coor, pic_coor),
                    Dfun=jacobian)[0]
 
    #返回拆解后参数，分别为内参矩阵，畸变矫正系数，每幅图对应外参矩阵
    return decompose_paramter_vector(P)
 
#把所有参数整合到一个数组内
def compose_paramter_vector(A, k, W):
    alpha = np.array([A[0, 0], A[1, 1], A[0, 1], A[0, 2], A[1, 2], k[0], k[1]])
    P = alpha
    for i in range(len(W)):
        R, t = (W[i])[:, :3], (W[i])[:, 3]

        w = np.append(R.reshape((1,-1)),t)
        P = np.append(P, w)
    return P
 
 
#分解参数集合，得到对应的内参，外参，畸变矫正系数
def decompose_paramter_vector(P):
    [alpha, beta, gamma, uc, vc, k0, k1] = P[0:7]
    A = np.array([[alpha, gamma, uc],
                  [0, beta, vc],
                  [0, 0, 1]])
    k = np.array([k0, k1])
    W = []
    M = (len(P) - 7) // 12
 
    for i in range(M):
        m = 7 + 12 * i
        R = (P[m:m+9]).reshape(3, -1)
        t = (P[m+9:m+12]).reshape(3, -1)

        #依次拼接每幅图的外参
        w = np.concatenate((R, t), axis=1)
        W.append(w)
 
    W = np.array(W)
    return A, k, W
 
 
#返回从真实世界坐标映射的图像坐标
def get_single_project_coor(A, W, k, coor):
    single_coor = np.array([coor[0], coor[1], coor[2], 1])
 
    coor_norm = np.dot(W, single_coor)
    coor_norm /= coor_norm[-1]
 
    r = np.sqrt(coor_norm[0] * coor_norm[0] + coor_norm[1] * coor_norm[1])

    uv = np.dot(np.dot(A, W), single_coor)
    uv /= uv[-1]
 
    #畸变
    u0 = uv[0]
    v0 = uv[1]
 
    uc = A[0, 2]
    vc = A[1, 2]
 
    u = u0 + (u0 - uc) * r**2 * k[0] + (u0 - uc) * r**4 * k[1]
    v = v0 + (v0 - vc) * r**2 * k[0] + (v0 - vc) * r**4 * k[1]
 
    return np.array([u, v])
 

#返回所有点的真实世界坐标映射到的图像坐标与真实图像坐标的残差
def value(P, org_W, X, Y_real):
    M = (len(P) - 7) // 12
    N = len(X[0])
    A = np.array([
        [P[0], P[2], P[3]],
        [0, P[1], P[4]],
        [0, 0, 1]
    ])
    Y = np.array([])
 
    for i in range(M):
        m = 7 + 12 * i

        #取出当前图像对应的外参
        t = (P[m+9:m+12]).reshape(3, -1)
        R = (P[m:m+9]).reshape(3, -1)
        W = np.concatenate((R, t), axis=1)
        
        #计算每幅图的坐标残差
        for j in range(N):
            Y = np.append(Y, get_single_project_coor(A, W, np.array([P[5], P[6]]), (X[i])[j]))
 
    error_Y  =  np.array(Y_real).reshape(-1) - Y
 
    return error_Y
 
 
#计算对应jacobian矩阵
def jacobian(P, WW, X, Y_real):
    M = (len(P) - 7) // 12
    N = len(X[0])
    K = len(P)
    A = np.array([
        [P[0], P[2], P[3]],
        [0, P[1], P[4]],
        [0, 0, 1]
    ])
 
    res = np.array([])
 
    for i in range(M):
        m = 7 + 12* i

        t = (P[m+9:m+12]).reshape(3, -1)
        R = (P[m:m+9]).reshape(3, -1)

        W = np.concatenate((R, t), axis=1)
 
        for j in range(N):
            res = np.append(res, get_single_project_coor(A, W, np.array([P[5], P[6]]), (X[i])[j]))
 
    #求得x, y方向对P[k]的偏导
    J = np.zeros((K, 2 * M * N))
    for k in range(K):
        J[k] = np.gradient(res, P[k])
 
    return J.T