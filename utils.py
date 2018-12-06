import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *

import scipy
import numpy as np

def read_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    if img_list == None:
        return None
        
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs
    
def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    return scipy.misc.imread(path + file_name).astype(np.float)
    # return scipy.misc.imread(path + file_name, mode='RGB')

def crop_sub_imgs_fn(x, is_random=True):
    # x = crop(x, wrg=hr_size, hrg=hr_size, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample_fn(x ,lr_size):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[lr_size, lr_size], interp='bicubic', mode=None)#tl.prepro.imresize
    x = x / (255. / 2.)
    x = x - 1.
    return x

def scale_fn(x):
    x = x / (255. / 2.)
    x = x - 1.
    return x