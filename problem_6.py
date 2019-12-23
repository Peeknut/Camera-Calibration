'''
Problem_6
ref:https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
    https://www.jianshu.com/p/23928a80fa0f
'''

import numpy as np
import cv2
import glob

# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 获取标定板角点的位置
objp = np.zeros((6*7,3), np.float32)  # 6*7的格子 此处参数根据使用棋盘格规格进行修改
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2) 
# 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y  


objpoints = [] # 存储3D点
imgpoints = [] # 存储2D点

images = glob.glob('left\\*.jpg') # 文件存储路径，存储需要标定的摄像头拍摄的棋盘格图片

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    # 寻找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    if ret == True:
        objpoints.append(objp)

        # 在原角点的基础上寻找亚像素角点
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        if corners2.any():
            imgpoints.append(corners2)
        else:
            imgpoints.append(corners)

        # 绘制角点并显示
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

print(len(imgpoints))
cv2.destroyAllWindows()

#标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("ret:",ret)
print("mtx:\n",mtx)
print("dist:\n",dist)
print("rvecs:\n",rvecs)
print("tvecs:\n",tvecs)

print("-------------------计算反向投影误差-----------------------") 
tot_error = 0
for i in range(len(objpoints)):  
    print(rvecs[i])
    img_points2, _ = cv2.projectPoints(objpoints[i],rvecs[i],tvecs[i],mtx,dist)  
    
    error = cv2.norm(imgpoints[i],img_points2, cv2.NORM_L2)/len(img_points2)  
    tot_error += error  
  
mean_error = tot_error/len(objpoints)  
print("total error: ", tot_error)  
print("mean error: ", mean_error ) 