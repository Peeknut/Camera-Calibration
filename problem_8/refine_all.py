#!usr/bin/env/ python
# _*_ coding:utf-8 _*_
 
import numpy as np
import math
from scipy import optimize as opt
 
#微调所有参数
def refinall_all_param(A, k, W, real_coor, pic_coor):
    #整合参数
    P_init = compose_paramter_vector(A, k, W)

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
 
        #旋转矩阵转换为一维向量形式
        zrou = to_rodrigues_vector(R)
 
        w = np.append(zrou, t)
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
    M = (len(P) - 7) // 6
 
    for i in range(M):
        m = 7 + 6 * i
        zrou = P[m:m+3]
        t = (P[m+3:m+6]).reshape(3, -1)
 
        #将旋转矩阵一维向量形式还原为矩阵形式
        R = to_rotation_matrix(zrou)
 
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
    M = (len(P) - 7) // 6
    N = len(X[0])
    A = np.array([
        [P[0], P[2], P[3]],
        [0, P[1], P[4]],
        [0, 0, 1]
    ])
    Y = np.array([])
 
    for i in range(M):
        m = 7 + 6 * i

        # 不用旋转矩阵的变换是因为会有精度损失
        '''
        #取出当前图像对应的外参
        w = P[m:m + 6]
        R = to_rotation_matrix(w[:3])
        t = w[3:].reshape(3, 1)
        W = np.concatenate((R, t), axis=1)
        '''
        W = org_W[i]
        #计算每幅图的坐标残差
        for j in range(N):
            Y = np.append(Y, get_single_project_coor(A, W, np.array([P[5], P[6]]), (X[i])[j]))
 
    error_Y  =  np.array(Y_real).reshape(-1) - Y
 
    return error_Y
 
 
#计算对应jacobian矩阵
def jacobian(P, WW, X, Y_real):
    M = (len(P) - 7) // 6
    N = len(X[0])
    K = len(P)
    A = np.array([
        [P[0], P[2], P[3]],
        [0, P[1], P[4]],
        [0, 0, 1]
    ])
 
    res = np.array([])
 
    for i in range(M):
        m = 7 + 6 * i
 
        w = P[m:m + 6]
        R = to_rotation_matrix(w[:3])
        t = w[3:].reshape(3, 1)
        W = np.concatenate((R, t), axis=1)
 
        for j in range(N):
            res = np.append(res, get_single_project_coor(A, W, np.array([P[5], P[6]]), (X[i])[j]))
 
    #求得x, y方向对P[k]的偏导
    J = np.zeros((K, 2 * M * N))
    for k in range(K):
        J[k] = np.gradient(res, P[k])
 
    return J.T
 
 
#将旋转矩阵分解为一个向量并返回，Rodrigues旋转向量与矩阵的变换,最后计算坐标时并未用到，因为会有精度损失
def to_rodrigues_vector(R):
    p = 0.5 * np.array([[R[2, 1] - R[1, 2]],
                        [R[0, 2] - R[2, 0]],
                        [R[1, 0] - R[0, 1]]])
    c = 0.5 * (np.trace(R) - 1)
 
    if np.linalg.norm(p) == 0:
        if c == 1:
            zrou = np.array([0, 0, 0])
        elif c == -1:
            R_plus = R + np.eye(3, dtype='float')
 
            norm_array = np.array([np.linalg.norm(R_plus[:, 0]),
                                   np.linalg.norm(R_plus[:, 1]),
                                   np.linalg.norm(R_plus[:, 2])])
            v = R_plus[:, np.where(norm_array == max(norm_array))]
            u = v / np.linalg.norm(v)
            if u[0] < 0 or (u[0] == 0 and u[1] < 0) or (u[0] == u[1] and u[0] == 0 and u[2] < 0):
                u = -u
            zrou = math.pi * u
        else:
            zrou = []
    else:
        u = p / np.linalg.norm(p)
        theata = math.atan2(np.linalg.norm(p), c)
        zrou = theata * u
 
    return zrou
 
 
#把旋转矩阵的一维向量形式还原为旋转矩阵并返回
def to_rotation_matrix(zrou):
    theta = np.linalg.norm(zrou)
    zrou_prime = zrou / theta
 
    W = np.array([[0, -zrou_prime[2], zrou_prime[1]],
                  [zrou_prime[2], 0, -zrou_prime[0]],
                  [-zrou_prime[1], zrou_prime[0], 0]])
    R = np.eye(3, dtype='float') + W * math.sin(theta) + np.dot(W, W) * (1 - math.cos(theta))
 
    return R