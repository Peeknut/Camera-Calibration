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

image_names.extend(glob.glob('left\\*.jpg'))
image_names.extend(glob.glob('right\\*.jpg'))

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

