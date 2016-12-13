import gzip
import os
import sys
import urllib
import csv
import matplotlib.image as mpimg
import code
import tensorflow.python.platform
import numpy as np
import tensorflow as tf

from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

from config import *

# Extract patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


def read_images(file_prefix, num_images, is_train):
    imgs = []
    filename_format = '{}{:03d}.png' if is_train else '{}{}.png'
    for i in range(1, num_images + 1):
        filename = filename_format.format(file_prefix, i)
        if os.path.isfile(filename):
            print('Loading {}'.format(filename))
            img = mpimg.imread(filename)
            if 'groundtruth' not in file_prefix:
                img = rgb_to_hsv(img)
            imgs.append(img)
        else:
            print('File {} does not exist'.format(filename))
    return imgs


def extract_patches(file_prefix, num_images, is_train):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = read_images(file_prefix, num_images, is_train)

    img_patches = [img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE) for img in imgs]
    data = [patch for patches in img_patches for patch in patches]

    return np.asarray(data)


# Extract label images
def extract_labels(file_prefix, num_images, is_train):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = read_images(file_prefix, num_images, is_train)

    gt_patches = [img_crop(gt_img, IMG_PATCH_SIZE, IMG_PATCH_SIZE) for gt_img in gt_imgs]
    data = np.asarray([gt_patch for patches in gt_patches for gt_patch in patches])
    labels = np.asarray([value_to_class(np.mean(d)) for d in data])

    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)


# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:
        return [1, 0]
    else:
        return [0, 1]

