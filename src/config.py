NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAIN_START = 1
TRAIN_SIZE = 100
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16 # 64
NUM_EPOCHS = 10
RECORDING_STEP = 1000
TRAIN_FORMAT = 'data/training/images/satImage_{:03d}.png'
TEST_FORMAT = 'data/test_set_images/test_{}.png'
TEST_START = 1
TEST_SIZE = 50
TRAIN_LABEL_FORMAT = 'data/training/groundtruth/satImage_{:03d}.png'
TRAIN_MODE = False
RESTORE_MODEL = True # If True, restore existing model instead of training a new one

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16
IMG_STRIDE_SIZE = 4

IMAGE_ROTATE = False
IMAGE_HSV_RANDOM = True
BALANCE_DATA = False
