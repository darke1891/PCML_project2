'''
Read functions for testing
When we get data for training, we use this file
'''

import numpy as np
import random

from config import *
from basic_read import extract_patches, extract_labels


def extract_train():
    train_data = extract_patches(TRAIN_FORMAT, TRAIN_START, TRAIN_SIZE, True)
    train_labels = extract_labels(TRAIN_LABEL_FORMAT, TRAIN_START,
                                  TRAIN_SIZE, True)
    return train_data, train_labels
