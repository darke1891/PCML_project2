import numpy as np

from config import *
from basic_read import extract_labels, read_images, img_to_patches

def extract_test_labels():
    labels = extract_labels(TRAIN_LABEL_FORMAT, TRAIN_START, TRAIN_SIZE, False)
    return labels

def extract_test_data(file_format, index0_images, num_images):
    images = read_images(file_format, index0_images, num_images)
    img_patches = [np.asarray(img_to_patches(img, False)) for img in images]
    test_data = [[images[i], img_patches[i]] for i in range(0, len(images))]
    return test_data
