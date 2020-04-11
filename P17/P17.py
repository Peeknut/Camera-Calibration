import numpy as np
import cv2


#读取左图并转为单通道灰度图
imgL = cv2.imread('left00.png')
imgLG = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)


#读取右图并转为单通道灰度图
imgR = cv2.imread('right00.png')
imgRG = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

#SGBM函数参数 
numberOfDisparities = ((int)(640/8) + 15) & -16 # 640为图片宽度



sgbmWinSize = 5
cn = 2 # 通道数，这里是灰度图
P1 = 8 * cn * sgbmWinSize ** 2
P2 = 4 * P1
blockSize = sgbmWinSize
minDisparity = 0
numDisparities = numberOfDisparities + minDisparity
uniquenessRatio = 10
speckleWindowSize = 100
speckleRange = 32 
disp12MaxDiff = 1 
preFilterCap = 63 

#计算视差
stereo = cv2.StereoSGBM_create(minDisparity=minDisparity, numDisparities=numDisparities, blockSize=blockSize,
                               uniquenessRatio=uniquenessRatio, speckleRange=speckleRange,
                               speckleWindowSize=speckleWindowSize, disp12MaxDiff=disp12MaxDiff, P1=P1, P2=P2,preFilterCap=preFilterCap)
disp = stereo.compute(imgLG, imgRG).astype(np.float32) / 16.0 # 除以16得到真实的视差

#转换为单通道图片
disp = cv2.normalize(disp, disp, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

#显示并输出图像
cv2.imshow("disp.jpg", disp)
cv2.imwrite("disp.jpg", disp)

#等待
cv2.waitKey()

#退出
cv2.destroyAllWindows()

