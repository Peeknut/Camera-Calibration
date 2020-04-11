import numpy as np
import cv2

imgL = cv2.imread('left.png')
imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgR = cv2.imread('right.png')
imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)


numDisparities = ((int)(640/8) + 15) & -16#640为图片宽度
SADWindowSize = 5

speckleWindowSize = 100 # 默认0
speckleRange = 32 # 默认0
disp12MaxDiff = 1 # 默认-1
# 默认值
minDisparity = 0
preFilterSize = 9 
preFilterCap = 31 
textureThreshold = 10
uniquenessRatio = 10

stereo = cv2.StereoBM_create(numDisparities = numDisparities, blockSize=SADWindowSize)

disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0 # 除以16得到真实视差值

#转换为单通道图片
disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite("disp.png", disp)
cv2.imshow("disp", disp)

cv2.waitKey(1000)
cv2.destroyAllWindows()