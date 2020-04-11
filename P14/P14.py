import numpy as np
import cv2
import glob

# 0.基本配置
show_corners = False

image_number = 13
board_size = (9, 6)  # 角点数量
square_Size = 20

image_lists = []  # 存储获取到的图像
image_points = []  # 存储图像的点

# 1.读图,找角点
image_names = []

image_names.extend(glob.glob('..\\left\\*.jpg'))
image_names.extend(glob.glob('..\\right\\*.jpg'))

for image_name in image_names:
    image = cv2.imread(image_name)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, board_size)  # 粗查找角点
    if not found:
        print("ERROR(no corners):" + image_name)


    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01) # 表示迭代停止的条件
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)  # 精定位角点
    image_points.append(corners.reshape(-1, 2))
    image_lists.append(image)

# 2. 构建标定板的点坐标，objectPoints
object_points = np.zeros((np.prod(board_size), 3), np.float32)
object_points[:, :2] = np.indices(board_size).T.reshape(-1, 2)
object_points *= square_Size # 20
object_points = [object_points] * image_number


# 3. 分别得到两个相机的初始CameraMatrix
h, w, _= image_lists[0].shape
camera_matrix = list()

camera_matrix.append(cv2.initCameraMatrix2D(object_points, image_points[:image_number], (w, h), 0))
camera_matrix.append(cv2.initCameraMatrix2D(object_points, image_points[image_number:], (w, h), 0))


# 4. 双目视觉进行标定
term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
    cv2.stereoCalibrate(object_points, image_points[:image_number], image_points[image_number:], camera_matrix[0],
                        None, camera_matrix[1], None, (w, h),
                        flags=cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_USE_INTRINSIC_GUESS |
                              cv2.CALIB_SAME_FOCAL_LENGTH | cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5,
                        criteria=term)

print("retval", retval)
print("cameraMatrix1:", cameraMatrix1)
print("distCoeffs1:", distCoeffs1) # 畸变矩阵
print("cameraMatrix2:", cameraMatrix2)
print("distCoeffs2:", distCoeffs2)
print("R:", R)
print("T:", T)
print("E:", E)
print("F:", F)


# 5. 标定精度的衡量
err = 0
cntPoints = 0
npt = len(image_points[0])
for i in range(image_number):

    newimgpoints1 = cv2.undistortPoints(image_points[i], cameraMatrix1, distCoeffs1, 0, cameraMatrix1)
    newimgpoints2 = cv2.undistortPoints(image_points[image_number+i], cameraMatrix2, distCoeffs2, 0, cameraMatrix2)
    line1 = cv2.computeCorrespondEpilines(newimgpoints1, 1, F) # 为一幅图像中的点计算其在另一幅图中对应的对极线
    line2 = cv2.computeCorrespondEpilines(newimgpoints2, 2, F)
    
    for j in range(npt):

        terr = np.abs(newimgpoints1[j][0][0] * line2[j][0][0] +
                    newimgpoints1[j][0][1] *  line2[j][0][1] + line2[j][0][2]) / np.sqrt(line2[j][0][0] ** 2 + line2[j][0][1] ** 2)
        terr += np.abs(newimgpoints2[j][0][0] * line1[j][0][0] +
                    newimgpoints2[j][0][1] *  line1[j][0][1] + line1[j][0][2]) / np.sqrt(line1[j][0][0] ** 2 + line1[j][0][1] ** 2)
        err += terr
    cntPoints += npt
print("average epipolar err = ", err/cntPoints)


# 7. 矫正图像，是否完成了极线矫正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = \
    cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (w, h), R, T)

map1_1, map1_2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (w, h), cv2.CV_16SC2)
map2_1, map2_2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (w, h), cv2.CV_16SC2)

for i in range(image_number):
    result1 = cv2.remap(image_lists[i], map1_1, map1_2, cv2.INTER_LINEAR)
    result2 = cv2.remap(image_lists[image_number+i], map2_1, map2_2, cv2.INTER_LINEAR)

    result = np.concatenate((result1, result2), axis=1)
    for j in range(15):
        cv2.line(result, (0, 50 * (j + 1)), (2 * w, 50 * (j + 1)), (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.imwrite("./result/rec%02d.png"%i, result)

