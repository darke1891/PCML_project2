NUM_THREADS = 16
MODEL_DIR = '/tmp/mnist-baoge'
NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAIN_SIZE = 80
TRAIN_START = 1
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = None  # Set to None for random seed.
BATCH_SIZE = 16 # 64
NUM_EPOCHS = 10
RECORDING_STEP = 1000
TRAIN_FORMAT = 'data/training/images/satImage_{:03d}.png'
TEST_FORMAT = 'data/test_set_images/test_{}.png'
TEST_START = 1
TEST_SIZE = 50
TRAIN_LABEL_FORMAT = 'data/training/groundtruth/satImage_{:03d}.png'
TRAIN_MODEL = False # if it's training(including doing predictions on remaining train images) 
RESTORE_MODEL = True # restore model from backup  
TEST = True # do predictions with test_set_images
TEST_SERVER = True
TEST_PATCH_SIZE = 19 * 19 * 64

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16
IMG_STRIDE_SIZE = 2
PADDING = int((IMG_PATCH_SIZE - IMG_STRIDE_SIZE) / 2)

IMAGE_ROTATE = False
IMAGE_HSV_RANDOM = False
BALANCE_DATA = False
ADAM = True
ADAM_INITIAL_RATE = 0.001
MOMENTUM_INITIAL_RATE = 0.01
MOMENTUM_MOMENTUM = 0.0
DROPOUT = False

WEIGHT_STD = 0.01


def print_config(outf):
    outf.write("TRAIN_SIZE = {}\n".format(TRAIN_SIZE))
    outf.write("NUM_EPOCHS = {}\n".format(NUM_EPOCHS))
    outf.write("BATCH_SIZE = {}\n".format(BATCH_SIZE))
    outf.write("IMG_PATCH_SIZE = {}\n".format(IMG_PATCH_SIZE))
    outf.write("IMG_STRIDE_SIZE = {}\n".format(IMG_STRIDE_SIZE))
    outf.write("IMAGE_ROTATE = {}\n".format(IMAGE_ROTATE))
    outf.write("IMAGE_HSV_RANDOM = {}\n".format(IMAGE_HSV_RANDOM))
    outf.write("BALANCE_DATA = {}\n".format(BALANCE_DATA))
    outf.write("DROPOUT = {}\n".format(DROPOUT))
    outf.write("WEIGHT_STD = {}\n".format(WEIGHT_STD))
    outf.write("ADAM = {}\n".format(ADAM))
    if ADAM:
        outf.write("ADAM_INITIAL_RATE = {}\n".format(ADAM_INITIAL_RATE))
    else:
        outf.write("MOMENTUM_INITIAL_RATE = {}\n".format(MOMENTUM_INITIAL_RATE))
        outf.write("MOMENTUM_MOMENTUM = {}\n".format(MOMENTUM_MOMENTUM))
    outf.write("\n")
