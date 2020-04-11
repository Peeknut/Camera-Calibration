import torch.utils.data as data
import random
from PIL import Image
from . import preprocess
# import preprocess
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def trim_image(img, img_height, img_width):
    """Trim image to remove pixels on the right and bottom.

    Args:
        img (numpy.ndarray): input image.
        img_height (int): desired image height.
        img_width (int): desired image width.

    Returns:
        (numpy.ndarray): trimmed image.

    """
    return img[0:img_height, 0:img_width]

def disparity_loader(image_path, img_height, img_width):
    """Load disparity image as numpy array.

    Args:
        image_path (str): path to disparity image.
        img_height (int): desired height of output image (excess trimmed).
        img_width (int): desired width of output image (excess trimmed).

    Returns:
        disp_img (numpy.ndarray): disparity image array as tensor.

    """
    disp_img = np.array(Image.open(image_path)).astype('float64')
    disp_img = trim_image(disp_img, img_height, img_width)
    disp_img /= 256

    return disp_img


class myImageFloder(data.Dataset):
    def __init__(self,
                 left,
                 right,
                 left_disparity,
                 training,
                 normalize,
                 loader=default_loader,
                 dploader=disparity_loader):
        '''
        参数：
            left: 左图路径列表
            train：true/false，是否进行训练的标志
            normalize：值为{'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]}
            loader：左图、右图以RGB形式读入函数
            dploader：视差图的读入函数

        '''
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.normalize = normalize

    def __getitem__(self, index):
        
        left = self.left[index]
        
        right = self.right[index]
        disp_L = self.disp_L[index]
        left_img = self.loader(left)
        right_img = self.loader(right)

        processed = preprocess.get_transform(
            augment=False, normalize=self.normalize)
        # 对图像进行归一化，范围在[0.0,1.0]
        left_img = processed(left_img)
        right_img = processed(right_img)


        h = left_img.size()[1]
        w = left_img.size()[2]
        dataL = self.dploader(disp_L, h, w)
        
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)

        return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
# if __name__ == '__main__':
#     path = '/media/lxy/sdd1/stereo_coderesource/dataset_nie/SceneFlowData/frames_cleanpass/flyingthings3d_disparity/TRAIN/A/0024/left/0011.pfm'
#     res = disparity_loader(path)
#     print(res.shape)
