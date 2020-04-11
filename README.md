# Camera-Calibration
### 运行环境
  * Win10
  * Python 3.7.5
  * opencv 4.1.2
### 文件说明
  * problem_8: 只进行了(A, R, t, k)的优化，并且优化中使用旋转矩阵R转换为旋转向量的方式
  * problem_8_amended: 进行(A, R, t)、(A, R, t, k)两次优化，并且优化中R不进行旋转向量的转换
### 引用说明
* problem_6.py、problem_7.py:
  + [1]https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
  + [2]https://www.jianshu.com/p/23928a80fa0f
* problem_8、problem_8_amended:
  + https://blog.csdn.net/qq_40369926/article/details/89251296#2.homography.py
* problem_12、problem_14：
  + https://www.jianshu.com/p/be990ccb9296
* problem_17：
  + https://blog.csdn.net/weixin_43459369/article/details/93623585
* problem_20：
  + https://github.com/jasonlinuxzhang/sgbm_cuda
* problem_21：
  + https://github.com/hjimce/Depth-Map-Prediction
* problem_22：
  + https://github.com/amrit110/stereo_matching
* problem_23：
  + https://github.com/meteorshowers/StereoNet-ActiveStereoNet
