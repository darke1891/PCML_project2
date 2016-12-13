NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 10
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16 # 64
NUM_EPOCHS = 2
RECORDING_STEP = 1000
TRAIN_PREFIX = 'data/training/images/satImage_'
TEST_PREFIX = 'data/test_set_images/test_'
TEST_SIZE = 50
TRAIN_LABEL_PREFIX = 'data/training/groundtruth/satImage_'
TRAIN_MODE = True
RESTORE_MODEL = False # If True, restore existing model instead of training a new one

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16

