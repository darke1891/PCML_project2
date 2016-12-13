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

def extract_train():
    train_data = extract_patches(TRAIN_PREFIX, TRAINING_SIZE, True)
    train_labels = extract_labels(TRAIN_LABEL_PREFIX, TRAINING_SIZE, True)
    return train_data, train_labels