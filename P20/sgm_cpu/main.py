from os.path import join, isfile
import time
import cv2
import numpy as np
from PIL import Image
import os
import os.path
import logger

SHRT_MAX = 32760
INT_MAX = SHRT_MAX * 10
WARP_SIZE = 32

MAX_DISPARITY = 96 #必须是WARP_SIZE的整数倍
INVALID_DISP_SCALED = -16
DISP_SCALE = 16 # 为了保存视差小数部分，所以乘上了这个数
DISP_SHIFT = 4
PATH_MAX = 128

# 误差计算参数（其值是根据devkit）
ABS_THRESH = 3.0
REL_THRESH = 0.05

has_gt = True # 是否计算误差
flag_write_files = True # 是否将计算的视差图结果保存

TAB_OFS = 256*4
TAB_SIZE = 256 + TAB_OFS*2
MAX_IMG_SIZE = (375, 1242) # 375 * 1242
MAX_IMG_DISPARITY_SIZE = (375, 1242, MAX_DISPARITY)


class SGM_PARAMS(object):
	def __init__(self, preFilterCap, BlockSize, uniquenessRatio, disp12MaxDiff, WindowSize):
		self.preFilterCap = preFilterCap
		self.BlockSize = BlockSize
		self.P1 = 8 * BlockSize * BlockSize
		self.P2 = 32 * BlockSize * BlockSize
		self.uniquenessRatio = uniquenessRatio
		self.disp12MaxDiff = disp12MaxDiff
		self.WindowSize = WindowSize
params = SGM_PARAMS(preFilterCap=63, BlockSize=9, uniquenessRatio=1, disp12MaxDiff=15, WindowSize=3)


d_clibTab = np.zeros(TAB_SIZE, dtype=np.uint8)

# ------------------cost的参数-----------------
d_imgleft_data = np.zeros(MAX_IMG_SIZE, dtype=np.uint8)
d_imgright_data = np.zeros(MAX_IMG_SIZE, dtype=np.uint8)

d_imgleft_grad = np.zeros(MAX_IMG_SIZE, dtype=np.uint8)
d_imgright_grad = np.zeros(MAX_IMG_SIZE, dtype=np.uint8)

d_pixDiff = np.zeros(MAX_IMG_DISPARITY_SIZE, dtype=np.int16)
d_hsum = np.zeros(MAX_IMG_DISPARITY_SIZE, dtype=np.int16)
d_cost = np.zeros(MAX_IMG_DISPARITY_SIZE, dtype=np.int16)

# -----------------aggregation的参数-----
d_sp = np.zeros(MAX_IMG_DISPARITY_SIZE, dtype=int)

# ----------------视差计算 & 视差优化的参数--------
d_disp = np.zeros(MAX_IMG_SIZE, dtype=np.int16)
d_mins = np.zeros(MAX_IMG_SIZE, dtype=int)
d_raw_disp = np.zeros(MAX_IMG_SIZE, dtype=np.int16)

disp2 = np.zeros(MAX_IMG_SIZE, dtype=np.int16)
disp2cost = np.zeros(MAX_IMG_SIZE, dtype=int)

d_outdisp = np.zeros(MAX_IMG_SIZE, dtype=np.int16)
disparity_uint16 = np.zeros(MAX_IMG_SIZE, dtype=np.uint16)


def fill_tab(TAB_SIZE, TAB_OFS, ftzero):
	for k in range(TAB_SIZE):
		d_clibTab[k] = min(max(k-TAB_OFS, -ftzero), ftzero) + ftzero
	# print("-------------------fill_tab finish! get the d_clibTab:----------------------")

def init():
	fill_tab(TAB_SIZE, TAB_OFS, params.preFilterCap)

# -------------视差计算 & 视差优化函数---------------
def MedianFilter(rows, cols, WindowSize):
	'''
	作用：去除噪声--中值滤波
	'''
	half = int(WindowSize / 2)
	mid_idx = int(WindowSize * WindowSize / 2)

	for row in range(0, rows):
		for col in range(0, cols):
			if (row>=half and row<rows-half and col>=half and col<cols-half):
				window = d_disp[row-half: row+half+1, col-half: col+half+1]
				window = np.ravel(window)
				window.sort()
				d_outdisp[row][col] = window[mid_idx]
			else:
				d_outdisp[row][col] = d_disp[row][col]

def lrcheck(disp12MaxDiff, rows, cols):
	'''
	作用：剔除错误匹配——左右一致性法
	'''
	for row in range(0, rows):
		for col in range(0, cols):
			disp2cost[row][col] = INT_MAX
			disp2[row][col] = INVALID_DISP_SCALED
		# 遍历每一列
		for col in range(cols-1, MAX_DISPARITY-1, -1):
			d = d_raw_disp[row][col]
			if (d == INVALID_DISP_SCALED):
				pass
			else:
				_x2 = col - d # 左图某列减去视差d之后的列
				if (disp2cost[row][_x2] > d_mins[row][col]):
					disp2cost[row][_x2] = d_mins[row][col]
					disp2[row][_x2] = d

		for col in range(MAX_DISPARITY, cols):
			d1 = d_disp[row][col]
			if (d1 == INVALID_DISP_SCALED):
				pass
			else:
				_d = int(d1/DISP_SCALE)
				d_ = int((d1 + DISP_SCALE -1) / DISP_SCALE)
				_x = col - _d
				x_ = col - d_
				if (0<=_x and _x<cols and disp2[row][_x]>=0 and abs(disp2[row][_x]-_d)>disp12MaxDiff and \
					0<=x_ and x_<cols and disp2[row][x_]>=0 and abs(disp2[row][x_]-d_>disp12MaxDiff)):
					d_disp[row][col] = INVALID_DISP_SCALED
	# print("----------------------lrcheck() finished!!!!! get d_disp-----------------------")

def get_disparity(uniquenessRatio, rows, cols):
	'''
	作用：根据路径代价聚合矩阵，得到处理之后的最小视差矩阵d_disp（提高精度后）
	'''
	for row in range(0, rows):
		for col in range(MAX_DISPARITY, cols):
			bestDisp = -1
			d = 0
			
			d_disp[row][col] = INVALID_DISP_SCALED
			d_mins[row][col] = INT_MAX
			d_raw_disp[row][col] = INVALID_DISP_SCALED

			minS = np.min(d_sp[row][col])
			bestDisp = np.where(d_sp[row][col]==minS)
			bestDisp = bestDisp[0][0]

			d_mins[row][col] = minS

			# 唯一性检查
			flag = False
			for d in range(0, MAX_DISPARITY):
				if ((abs(bestDisp-d)>1) and (d_sp[row][col][d]*(100-uniquenessRatio) < minS*100)):
					flag = True
					break
			if (flag): # 说明求得的视差不对，该点视差值取-16
				pass
			else:
				d_raw_disp[row][col] = bestDisp

				# 提高视差精度--采用二次曲线内插法
				if ((bestDisp == 0) or (bestDisp == MAX_DISPARITY-1)):
					bestDisp = bestDisp * DISP_SCALE # DISP_SCALE=16
				else:
					denom2 = max(d_sp[row][col][bestDisp-1] + d_sp[row][col][bestDisp+1] - 2*minS, 1)
					bestDisp = bestDisp * DISP_SCALE + ((d_sp[row][col][bestDisp-1] - d_sp[row][col][bestDisp+1]) * DISP_SCALE \
								+ denom2) / (denom2 * 2)
				d_disp[row][col] = bestDisp
	# print("-------------------------------------get_disparity() finished!!!!!!!!!  get d_cost--------------------")

# -------------aggregation函数-----------------------
def cost_aggregation_rl_ud(p1, p2, rows, cols):
	'''
	作用：聚合方向：右上--》左下

	'''
	lr_pre = np.zeros(MAX_DISPARITY+2, dtype=int)
	lr_pre_temp = np.zeros(MAX_DISPARITY+2, dtype=int)

	# 遍历经过(0, col)，且斜率k=1直线上的点（上半部分的点）
	for col in range(MAX_DISPARITY, cols):
		# 初始化
		delta = p2
		lr_pre[0] = lr_pre[MAX_DISPARITY+1] = lr_pre_temp[0] = lr_pre_temp[MAX_DISPARITY+1] = INT_MAX
		for d in range(1, MAX_DISPARITY+1):
			lr_pre[d] = lr_pre_temp[d] = 0
		# 遍历
		tp_col = col
		len = min(rows, col-MAX_DISPARITY+1)
		for row in range(0, len):
			minlr = INT_MAX
			for d in range(0, MAX_DISPARITY):
				lr_pre_temp[d+1] = d_cost[row][tp_col][d] + min(lr_pre[d+1], lr_pre[d]+p1, lr_pre[d+2]+p1, delta) - delta
				minlr = min(minlr, lr_pre_temp[d+1])
				d_sp[row][tp_col][d] += lr_pre_temp[d+1]
			delta = p2 + minlr

			pt = lr_pre
			lr_pre = lr_pre_temp
			lr_pre_temp = pt

			tp_col -= 1

	# 遍历经过(row, cols)，且斜率k=1直线上的点（下半部分的点）
	for row in range(1, rows):
		# 初始化
		delta = p2
		lr_pre[0] = lr_pre[MAX_DISPARITY+1] = lr_pre_temp[0] = lr_pre_temp[MAX_DISPARITY+1] = INT_MAX
		for d in range(1, MAX_DISPARITY+1):
			lr_pre[d] = lr_pre_temp[d] = 0
		# 遍历
		tp_row = row
		tp_col = cols - 1
		for i in range(0, rows-row):
			minlr = INT_MAX
			for d in range(0, MAX_DISPARITY):
				lr_pre_temp[d+1] =d_cost[tp_row][tp_col][d] + min(lr_pre[d+1], lr_pre[d]+p1, lr_pre[d+2]+p1, delta) - delta
				minlr = min(minlr, lr_pre_temp[d+1])
				d_sp[tp_row][tp_col][d] += lr_pre_temp[d+1]
			delta = p2 + minlr

			pt = lr_pre
			lr_pre = lr_pre_temp
			lr_pre_temp = pt

			tp_col -= 1
			tp_row += 1
	# print("-------------------------------cost_aggregation_rl_ud() finished!!!! get d_sp------------------")

def cost_aggregation_ud_lr(p1, p2, rows, cols):
	'''
	作用：聚合方向：左上--》右下
	'''
	lr_pre = np.zeros(MAX_DISPARITY+2, dtype=int)
	lr_pre_temp = np.zeros(MAX_DISPARITY+2, dtype=int)

	# 遍历经过(0, col)，且斜率k=-1直线上的点（上半部分的点）
	for col in range(MAX_DISPARITY, cols):
		# 初始化
		delta = p2
		lr_pre[0] = lr_pre[MAX_DISPARITY+1] = lr_pre_temp[0] = lr_pre_temp[MAX_DISPARITY+1] = INT_MAX
		for d in range(1, MAX_DISPARITY+1):
			lr_pre[d] = lr_pre_temp[d] = 0
		# 遍历
		tp_col = col
		len = min(rows, cols-col)
		for row in range(0, len):
			minlr = INT_MAX
			for d in range(0, MAX_DISPARITY): # 表示视差
				lr_pre_temp[d+1] = d_cost[row][tp_col][d] + min(lr_pre[d+1], lr_pre[d]+p1, lr_pre[d+2]+p1, delta) - delta
				minlr = min(minlr, lr_pre_temp[d+1])
				d_sp[row][tp_col][d] += lr_pre_temp[d+1]
			delta = p2 + minlr

			pt = lr_pre
			lr_pre = lr_pre_temp
			lr_pre_temp = pt

			tp_col += 1

	# 遍历经过(row, 0)，且斜率k=-1直线上的点（下半部分的点）
	for row in range(1, rows):
		# 初始化
		delta = p2
		lr_pre[0] = lr_pre[MAX_DISPARITY+1] = lr_pre_temp[0] = lr_pre_temp[MAX_DISPARITY+1] = INT_MAX
		for d in range(1, MAX_DISPARITY+1):
			lr_pre[d] = lr_pre_temp[d] = 0
		# 遍历
		tp_row = row
		tp_col = MAX_DISPARITY
		for i in range(0, rows-row):
			minlr = INT_MAX
			for d in range(0, MAX_DISPARITY):
				lr_pre_temp[d+1] =d_cost[tp_row][tp_col][d] + min(lr_pre[d+1], lr_pre[d]+p1, lr_pre[d+2]+p1, delta) - delta
				minlr = min(minlr, lr_pre_temp[d+1])
				d_sp[tp_row][tp_col][d] += lr_pre_temp[d+1]
			delta = p2 + minlr

			pt = lr_pre
			lr_pre = lr_pre_temp
			lr_pre_temp = pt

			tp_col += 1
			tp_row += 1
	# print("-------------------------------cost_aggregation_ud_lr() finished!!!! get d_sp------------------")

def cost_aggregation_du(p1, p2, rows, cols):
	lr_pre = np.zeros(MAX_DISPARITY+2, dtype=int)
	lr_pre_temp = np.zeros(MAX_DISPARITY+2, dtype=int)

	for col in range(MAX_DISPARITY, cols):
		# 初始化
		delta = p2
		lr_pre_temp[0] = lr_pre_temp[MAX_DISPARITY+1] = lr_pre[0] = lr_pre[MAX_DISPARITY+1] = INT_MAX
		for d in range(1, MAX_DISPARITY+1):
			lr_pre_temp[d] = lr_pre[d] = 0
		for row in range(rows-1, -1, -1):
			minlr = INT_MAX
			for d in range(0, MAX_DISPARITY):
				lr_pre_temp[d+1] = d_cost[row][col][d] + min(lr_pre[d+1], lr_pre[d]+p1, lr_pre[d+2]+p1, delta) - delta
				minlr = min(minlr, lr_pre_temp[d+1])
				d_sp[row][col][d] += lr_pre_temp[d+1]
			delta = p2 + minlr

			pt = lr_pre
			lr_pre = lr_pre_temp
			lr_pre_temp = pt

	print("-------------------------------cost_aggregation_du() finished!!!! get d_sp------------------")

def cost_aggregation_ud(p1, p2, rows, cols):
	'''
	作用：聚合方向：上--》下
	'''
	lr_pre = np.zeros(MAX_DISPARITY+2, dtype=int)
	lr_pre_temp = np.zeros(MAX_DISPARITY+2, dtype=int)

	for col in range(MAX_DISPARITY, cols):
		# 初始化
		delta = p2
		lr_pre_temp[0] = lr_pre_temp[MAX_DISPARITY+1] = lr_pre[0] = lr_pre[MAX_DISPARITY+1] = INT_MAX
		for d in range(1, MAX_DISPARITY+1):
			lr_pre_temp[d] = lr_pre[d] = 0
		for row in range(0, rows):
			minlr = INT_MAX
			for d in range(0, MAX_DISPARITY):
				lr_pre_temp[d+1] = d_cost[row][col][d] + min(lr_pre[d+1], lr_pre[d]+p1, lr_pre[d+2]+p1, delta) - delta
				minlr = min(minlr, lr_pre_temp[d+1])
				d_sp[row][col][d] += lr_pre_temp[d+1]
			delta = p2 + minlr

			pt = lr_pre
			lr_pre = lr_pre_temp
			lr_pre_temp = pt

	# print("-------------------------------cost_aggregation_ud() finished!!!! get d_sp------------------")

def cost_aggregation_rl(p1, p2, rows, cols):
	'''
	作用：聚合方向：右--》左
	'''
	lr_pre = np.zeros(MAX_DISPARITY+2, dtype=int)
	lr_pre_temp = np.zeros(MAX_DISPARITY+2, dtype=int)
	for row in range(0, rows):
		# 初始化
		delta = p2
		lr_pre[0] = lr_pre[MAX_DISPARITY+1] = lr_pre_temp[0] = lr_pre_temp[MAX_DISPARITY+1] = INT_MAX
		for d in range(1, MAX_DISPARITY+1):
			lr_pre_temp[d] = lr_pre[d] = 0
		# 遍历
		for col in range(cols-1, MAX_DISPARITY-1, -1):
			minlr = INT_MAX
			for d in range(0, MAX_DISPARITY):
				lr_pre_temp[d+1] = d_cost[row][col][d] + min(lr_pre[d+1], lr_pre[d]+p1, lr_pre[d+2]+p1, delta) - delta
				minlr = min(minlr, lr_pre_temp[d+1])
				d_sp[row][col][d] += lr_pre_temp[d+1]
			delta = p2 + minlr

			pt = lr_pre
			lr_pre = lr_pre_temp
			lr_pre_temp = pt
	# print("-------------------------------cost_aggregation_rl() finished!!!! get d_sp------------------")

def cost_aggregation_lr(p1, p2, rows, cols):
	'''
	作用：聚合方向：左--》右
		计算像素i在视差为d时的聚合，lr_pre[d+1]、lr_pre[d-1]、lr_pre[d]都是该点左边点的视差为d+1、d-1、d时的聚合值
		公式为：local_sp[d]=local_cost[d]+min(delta, lr_pre[d+1]+p1,lr_pre[d-1]+p1,lr_pre[d])-delta
	'''
	lr_pre = np.zeros(MAX_DISPARITY+2, dtype=int)
	lr_pre_temp = np.zeros(MAX_DISPARITY+2, dtype=int)

	for row in range(0, rows):
		delta = p2
		lr_pre[0] = lr_pre[MAX_DISPARITY+1] = lr_pre_temp[0] = lr_pre_temp[MAX_DISPARITY+1] = INT_MAX
		for d in range(1, MAX_DISPARITY+1):
			lr_pre_temp[d] = lr_pre[d] = 0
		# 遍历该行的每一列（MAX_DISPARITY, cols）
		for col in range(MAX_DISPARITY, cols):
			minlr = INT_MAX
			for d in range(0, MAX_DISPARITY): # 表示视差
				lr_pre_temp[d+1] = d_cost[row][col][d] + min(lr_pre[d+1], lr_pre[d]+p1, lr_pre[d+2]+p1, delta) - delta # 之前d_cost中都加了p2
				minlr = min(minlr, lr_pre_temp[d+1])
				d_sp[row][col][d] += lr_pre_temp[d+1]
			delta = p2 + minlr 
			
			pt = lr_pre
			lr_pre = lr_pre_temp
			lr_pre_temp = pt
	# print("-------------------------------cost_aggregation_lr() finished!!!! get d_sp------------------")

# ----------cost函数----------------
def get_cost(rows, cols, blocksize, p2):
	'''
	作用：某一列的第i行像素的d_cost值（以下式子是同一个视差计算的）
		d_cost[i] = p2+d_hsum[i-blocksize/2]+...+d_hsum[i]+...+d_hsum[i+blocksize/2]（上下聚合）,i的范围(0, rows)
	'''
	SH2 = int(blocksize / 2)
	for col in range(MAX_DISPARITY, cols):
		for now_disparity in range(0, MAX_DISPARITY):
			# 第0行的
			row = 0
			d_cost[row][col][now_disparity] = p2 + d_hsum[row][col][now_disparity] * (SH2+1)
			for i in range(1, SH2+1):
				d_cost[row][col][now_disparity] += d_hsum[row+i][col][now_disparity]

			#处理行数1-rows
			if (MAX_DISPARITY == col) :
				for row in range(1, rows):
					d_cost[row][col][now_disparity] = d_cost[row-1][col][now_disparity]
			else:
				for row in range(1, rows-SH2):
					tp = max(row-SH2-1, 0)
					h_sumSub = d_hsum[tp][col][now_disparity]
					h_sumAdd = d_hsum[row+SH2][col][now_disparity]
					d_cost[row][col][now_disparity] = d_cost[row-1][col][now_disparity] + h_sumAdd - h_sumSub
				for row in range(rows-SH2, rows):
					d_cost[row][col][now_disparity] = d_cost[row-1][col][now_disparity]
	# print("----------------------------------get_cost() finished!!! get d_cost------------------")

def get_hsum(rows, cols, blocksize):
	'''
	作用：某一行的第i个像素的d_hsum值（以下式子是同一个视差计算的）
		d_hsum[i] = d_pixel_diff[i-blocksize/2]+...+d_pixel_diff[i]+...+d_pixel_diff[i+blocksize/2](左右聚合),i的范围(MAX_DISPARITY, cols)
				---因为get_pixel_diff()所以从MAX_DISPARITY开始
	'''
	
	SW2 = int(blocksize / 2)
	
	for row in range(0, rows):
		for now_disparity in range(0, MAX_DISPARITY):
			# 对于某行row第MAX_DISPARITY列
			col = MAX_DISPARITY
			d_hsum[row][col][now_disparity] = d_pixDiff[row][col][now_disparity] * (SW2+1)
			for i in range(1, SW2+1):
				d_hsum[row][col][now_disparity] += d_pixDiff[row][col+i][now_disparity]


			# 遍历该行的第MAX_DISPARITY+1个到cols个像素
			for col in range(MAX_DISPARITY+1, cols):
				tp = min(col+SW2, cols-1)
				pixAdd = d_pixDiff[row][tp][now_disparity]
				tp = max(col-SW2-1, MAX_DISPARITY)
				pixSub = d_pixDiff[row][tp][now_disparity]
				d_hsum[row][col][now_disparity] = d_hsum[row][col-1][now_disparity] + pixAdd - pixSub
	# print("----------------------get_hsum() finished!!!! get d_hsum-----------------------")

def get_pixel_diff(rows, cols):
	d_imgright_buf = np.zeros((rows, cols, 3), dtype=np.int16)
	for row in range(0, rows): # 每一行
		# 右图
		for col in range(0, cols):
			v = int(d_imgright_grad[row][col]) 
			vl = (d_imgright_grad[row][col-1] + v) / 2 if col>0 else v
			vr = (d_imgright_grad[row][col+1] + v) / 2 if col<cols-1 else v

			v0 = min(vl, vr, v)
			v1 = max(vl, vr, v)

			d_imgright_buf[row][col][0] = v0
			d_imgright_buf[row][col][1] = v1
			d_imgright_buf[row][col][2] = v
		# 左图
		for col in range(MAX_DISPARITY, cols): # 每一列
			u = int(d_imgleft_grad[row][col])
			ul = (u + d_imgleft_grad[row][col-1]) / 2
			ur = (u + d_imgleft_grad[row][col+1]) / 2 if col<cols-1 else u

			u0 = min(ul, ur, u)
			u1 = max(ul, ur, u)

			for now_disparity in range(0, MAX_DISPARITY):
				c0 = max(0, u-d_imgright_buf[row][col-now_disparity][1], d_imgright_buf[row][col-now_disparity][0]-u)
				c1 = max(0, d_imgright_buf[row][col-now_disparity][2]-u1, u0-d_imgright_buf[row][col-now_disparity][2])
				d_pixDiff[row][col][now_disparity] += min(c0, c1)


	for row in range(0, rows): # 每一行
		# 右图
		for col in range(0, cols):
			v = int(d_imgright_data[row][col])
			vl = (d_imgright_data[row][col-1] + v) / 2 if col>0 else v
			vr = (d_imgright_data[row][col+1] + v) / 2 if col<cols-1 else v

			v0 = min(vl, vr, v)
			v1 = max(vl, vr, v)

			d_imgright_buf[row][col][0] = v0
			d_imgright_buf[row][col][1] = v1
			d_imgright_buf[row][col][2] = v
		# 左图
		for col in range(MAX_DISPARITY, cols): # 每一列
			u = int(d_imgleft_data[row][col])
			ul = (u + d_imgleft_data[row][col-1]) / 2
			ur = (u + d_imgleft_data[row][col+1]) / 2 if col<cols-1 else u

			u0 = min(ul, ur, u)
			u1 = max(ul, ur, u)

			for now_disparity in range(0, MAX_DISPARITY):
				c0 = max(0, u-d_imgright_buf[row][col-now_disparity][1], d_imgright_buf[row][col-now_disparity][0]-u)
				c1 = max(0, d_imgright_buf[row][col-now_disparity][2]-u1, u0-d_imgright_buf[row][col-now_disparity][2])

				pre_cost = d_pixDiff[row][col][now_disparity]
				d_pixDiff[row][col][now_disparity] =  pre_cost + int(min(c0, c1) / 4)
	# print("-------------------------get_pixel_diff() finished !  get d_pixel_diff-----------------")

def get_gradient(d_rows, d_cols):
	'''
	作用：转换d_imgleft_data为d_imgleft_grad，得到像素的变化梯度(每个像素的x导数)
	'''
	value = d_clibTab[TAB_OFS]

	for row in range(1, d_rows-1):
		d_imgleft_grad[row][0] = value
		d_imgright_grad[row][0] = value
		d_imgleft_data[row][0] = value
		d_imgright_data[row][0] = value

		d_imgleft_grad[row][d_cols - 1] = value
		d_imgright_grad[row][d_cols - 1] = value
		d_imgleft_data[row][d_cols - 1] = value
		d_imgright_data[row][d_cols - 1] = value
		for col in range(1, d_cols - 1):
			k = (int(d_imgleft_data[row][col+1]) - d_imgleft_data[row][col-1]) * 2 \
							+ d_imgleft_data[row-1][col+1] \
							- d_imgleft_data[row-1][col-1] \
							+ d_imgleft_data[row+1][col+1] \
							- d_imgleft_data[row+1][col-1]
			d_imgleft_grad[row][col] = d_clibTab[TAB_OFS+k]
			k = (int(d_imgright_data[row][col+1]) - d_imgright_data[row][col-1]) * 2 \
							+ d_imgright_data[row-1][col+1] \
							- d_imgright_data[row-1][col-1] \
							+ d_imgright_data[row+1][col+1] \
							- d_imgright_data[row+1][col-1]
			d_imgright_grad[row][col] = d_clibTab[TAB_OFS+k]
	# row==0
	row = 0
	d_imgleft_grad[row][0] = value
	d_imgright_grad[row][0] = value
	d_imgleft_data[row][0] = value
	d_imgright_data[row][0] = value

	d_imgleft_grad[row][d_cols - 1] = value
	d_imgright_grad[row][d_cols - 1] = value
	d_imgleft_data[row][d_cols - 1] = value
	d_imgright_data[row][d_cols - 1] = value
	for col in range(1, d_cols - 1):
		k = (int(d_imgleft_data[row][col+1]) - d_imgleft_data[row][col-1]) * 3 \
						+ d_imgleft_data[row+1][col+1] \
						- d_imgleft_data[row+1][col-1]
		d_imgleft_grad[row][col] = d_clibTab[TAB_OFS+k]
		k = (int(d_imgright_data[row][col+1]) - d_imgright_data[row][col-1]) * 3 \
						+ d_imgright_data[row+1][col+1] \
						- d_imgright_data[row+1][col-1]
		d_imgright_grad[row][col] = d_clibTab[TAB_OFS+k]
		
	# row==d_rows-1
	row = d_rows-1
	d_imgleft_grad[row][0] = value
	d_imgright_grad[row][0] = value
	d_imgleft_data[row][0] = value
	d_imgright_data[row][0] = value

	d_imgleft_grad[row][d_cols - 1] = value
	d_imgright_grad[row][d_cols - 1] = value
	d_imgleft_data[row][d_cols - 1] = value
	d_imgright_data[row][d_cols - 1] = value
	for col in range(1, d_cols - 1):
		k = (int(d_imgleft_data[row][col+1]) - d_imgleft_data[row][col-1]) * 3 \
						+ d_imgleft_data[row-1][col+1] \
						- d_imgleft_data[row-1][col-1] 
		d_imgleft_grad[row][col] = d_clibTab[TAB_OFS+k]
		k = (int(d_imgright_data[row][col+1]) - d_imgright_data[row][col-1]) * 3 \
						+ d_imgright_data[row-1][col+1] \
						- d_imgright_data[row-1][col-1] 
		d_imgright_grad[row][col] = d_clibTab[TAB_OFS+k]
	# print("-------------------get_gradient finish! get the d_imgleft_grad、d_imgright_grad:----------------------")
	
def compute_disparity(imgLG, imgRG):
	rows = imgLG.shape[0]
	cols = imgLG.shape[1]
	
	global d_imgleft_data
	global d_imgright_data
	d_imgleft_data = imgLG.copy()
	d_imgright_data = imgRG.copy()

	# **************************************计算cost**************************************-
	get_gradient(rows, cols)

	get_pixel_diff(rows, cols)	

	get_cost(rows, cols, params.BlockSize, params.P2)

	# ***************************路径聚合cost_aggregation:5个方向*****************************
	cost_aggregation_lr(params.P1, params.P2, rows, cols)

	# cost_aggregation_ud_lr(params.P1, params.P2, rows, cols)

	cost_aggregation_ud(params.P1, params.P2, rows, cols)

	# cost_aggregation_rl_ud(params.P1, params.P2, rows, cols)

	cost_aggregation_rl(params.P1, params.P2, rows, cols)

	cost_aggregation_du(params.P1, params.P2, rows, cols)
	# ***********************************得到视差（精度优化后的）*****************************************
	get_disparity(params.uniquenessRatio, rows, cols)

	# ***************************************视差优化*****************************************
	# 剔除错误匹配——左右一致性法
	lrcheck(params.disp12MaxDiff, rows, cols)

	MedianFilter(rows, cols, params.WindowSize)

	# *****************************************返回结果处理-----------------------------------
	for i in range(rows):
		for j in range(cols):
			if (d_outdisp[i][j] <= 0):
				disparity_uint16[i][j] = 0
			else:
				disparity_uint16[i][j] = d_outdisp[i][j] * (256 / DISP_SCALE)
	return disparity_uint16

def disparity_errors(estimation, gt_file_name):
	'''
	作用：计算误差
	'''
	gt_image = cv2.imread(gt_file_name, cv2.IMREAD_UNCHANGED)
	# if (gt_image == None):
	# 	print("-------read gt_image:", gt_file_name, "err----------------")
	# 	return -1,-1
	# if (gt_image.shape != estimation.shape):
	# 	print("------The shape of gt_image & estimation are not match-------------")
	# 	print("gt_image:", gt_image.shape)
	# 	print("estimation:", estimation.shape)
	# 	return -1, -1

	nlocal = 0
	nerrlocal = 0

	rows = gt_image.shape[0]
	cols = gt_image.shape[1]
	for i in range(rows):
		for j in range(MAX_DISPARITY, cols):
			gt = gt_image[i][j] # gt_file的像素
			if (gt > 0): # gt==0表示是非法点
				gt = gt / 256
				est = estimation[i][j] / 256

				err = abs(est - gt)
				ratio = err / gt
				if (err > ABS_THRESH and ratio > REL_THRESH):
					nerrlocal += 1
				nlocal += 1
	return nlocal, nerrlocal


def KITTI2015_dataloader(directory):
	left_fold = 'image_2/'
	right_fold = 'image_3/'
	disp_L = 'disp_occ_0/'
	res = 'result/'

	image = [img for img in os.listdir(directory+left_fold) if img.find('_10') > -1]


	left_img_name_list = [directory+left_fold+img for img in image]
	right_img_name_list = [directory+right_fold+img for img in image]
	gt_file_list = [directory+disp_L+img for img in image]
	dis_file_list = [directory+res+img for img in image]

	return left_img_name_list, right_img_name_list, gt_file_list, dis_file_list

def main():
	
	if(MAX_DISPARITY % WARP_SIZE != 0):
		print("configuration wrong\nMAX_DISPARITY must be divided by WARP_SIZE")
		return -1

	directory = "KITTI2015_part_data/"

	# 读取图片路径列表
	left_img_name_list, right_img_name_list, gt_file_list, dis_file_list= KITTI2015_dataloader(directory)
	times = []
	n = 0
	n_err = 0
	for i in range(len(left_img_name_list)):

		imgL = cv2.imread(left_img_name_list[i])
		imgLG = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
		imgR = cv2.imread(right_img_name_list[i])
		imgRG = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
		
		start_time = time.time()
		disparity_img = compute_disparity(imgLG, imgRG)
		end_time = time.time()
		times.append(end_time - start_time)

		if (has_gt):
			tp_n, tp_n_err = disparity_errors(disparity_img, gt_file_list[i])
			n += tp_n
			n_err += tp_n_err
		if (flag_write_files):
			cv2.imwrite(dis_file_list[i], disparity_img)

	# 输出
	mean_time = sum(times) / len(times)

	log = logger.setup_logger('result_speed_err.log')
	log.info("speed:")
	log.info("It took an average of "+ str(mean_time)+ " miliseconds, "+ str(1000/mean_time)+" fps")
	
	if (has_gt):
		log.info("error:"+ str(n_err/n))

if __name__ == "__main__":
	init()
	main()
