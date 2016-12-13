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
from basic_read import extract_patches, extract_labels

def rotate_data(input_data, input_labels):
    n = input_data.shape[0]
    data = np.repeat(input_data, 4, axis=0)
    labels = np.repeat(input_labels, 4, axis=0)
    for i in range(0, n):
        for j in range(1, 4):
            index = i * 4 + j
            data[index, :] = np.rot90(data[index, :])
    return data, labels


def extract_train():
    train_data = extract_patches(TRAIN_PREFIX, TRAIN_START, TRAIN_SIZE, True)
    train_labels = extract_labels(TRAIN_LABEL_PREFIX, TRAIN_START, TRAIN_SIZE, True)
    if IMAGE_ROTATE:
        train_data, train_labels = rotate_data(train_data, train_labels)
    return train_data, train_labels