NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAIN_SIZE = 80
TRAIN_START = 100 - TRAIN_SIZE + 1
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = None  # Set to None for random seed.
BATCH_SIZE = 16 # 64
NUM_EPOCHS = 1
RECORDING_STEP = 1000
TRAIN_FORMAT = 'data/training/images/satImage_{:03d}.png'
TEST_FORMAT = 'data/test_set_images/test_{}.png'
TEST_SIZE = 50
TRAIN_LABEL_FORMAT = 'data/training/groundtruth/satImage_{:03d}.png'
TRAIN_MODEL = True # if it's training(including doing predictions on remaining train images) 
RESTORE_MODEL = False # restore model from backup  
TEST = False # do predictions with test_set_images

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16
IMG_STRIDE_SIZE = 16
PADDING = int((IMG_PATCH_SIZE - IMG_STRIDE_SIZE) / 2)

IMAGE_ROTATE = False
IMAGE_HSV_RANDOM = False
BALANCE_DATA = False
ADAM = True
ADAM_INITIAL_RATE = 0.001
MOMENTUM_INITIAL_RATE = 0.01
MOMENTUM_MOMENTUM = 0.0
DROPOUT = True
