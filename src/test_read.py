import numpy as np

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
