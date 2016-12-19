import numpy as np
import random

from config import *
from basic_read import extract_patches, extract_labels

def rotate_data(input_data, input_labels):
    n = input_data.shape[0]
    data = np.repeat(input_data, 4, axis=0)
    labels = np.repeat(input_labels, 4, axis=0)
    for i in range(0, n):
        for j in range(1, 4):
            index = i * 4 + j
            data[index, :] = np.rot90(data[index, :], j)
    return data, labels


def random_hsv_data(input_data, input_labels):
    scale = 4
    data = np.repeat(input_data, scale, axis=0)
    labels = np.repeat(input_labels, scale, axis=0)
    for i in range(0, data.shape[0]):
        if i % scale != 0:
            data[i, :, :, 1] *= random.uniform(0.8, 1.2)
            data[i, :, :, 2] *= random.uniform(0.8, 1.2)
    return data, labels


def extract_train():
    train_data = extract_patches(TRAIN_FORMAT, TRAIN_START, TRAIN_SIZE, True)
    train_labels = extract_labels(TRAIN_LABEL_FORMAT, TRAIN_START, TRAIN_SIZE, True)
    if IMAGE_ROTATE:
        train_data, train_labels = rotate_data(train_data, train_labels)
    if IMAGE_HSV_RANDOM:
        train_data, train_labels = random_hsv_data(train_data, train_labels)
    return train_data, train_labels
