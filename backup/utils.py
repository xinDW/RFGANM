import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import numpy as np

hr_size = config.TRAIN.hr_size
lr_size = config.TRAIN.lr_size

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    return scipy.misc.imread(path + file_name).astype(np.float)
    # return scipy.misc.imread(path + file_name, mode='RGB')

def crop_sub_imgs_fn(x, is_random=True):
    # x = crop(x, wrg=hr_size, hrg=hr_size, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[lr_size, lr_size], interp='bicubic', mode=None)#tl.prepro.imresize
    x = x / (255. / 2.)
    x = x - 1.
    return x

def scale_fn(x):
    x = x / (255. / 2.)
    x = x - 1.
    return x