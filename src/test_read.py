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
from basic_read import extract_labels, read_images, img_to_patches

def extract_test_labels():
    labels = extract_labels(TRAIN_LABEL_PREFIX, TRAIN_START, TRAIN_SIZE, True)
    return labels

def extract_test_data(file_prefix, index0_images, num_images, is_train):
    images = read_images(file_prefix, index0_images, num_images, is_train)
    img_patches = [np.asarray(img_to_patches(img)) for img in images]
    test_data = [[images[i], img_patches[i]] for i in range(0, len(images))]
    return test_data