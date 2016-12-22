'''
Config file for the whole project.
Most parameters that we need to change in the proejct will be here.

To pass Pep8 E501, we have to make lines short enough.
So, sometimes we put comment under a line but not on the right of the line.
'''


NUM_THREADS = 32  # Max number of threads that TensorFlow can use.
MODEL_DIR = 'final2'  # The folder contains model
NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2  # Road and background
TRAIN_START = 1
'''
Train set is from satImage_001.png to satImage_(TRAIN_SIZE).png
Always set TRAIN_START 1, otherwise validation set will be uncorrect.
We do this because it's really slow to train the model
and we can't run multi-fold test.
'''
TRAIN_SIZE = 100
'''
Size of train set. Other data will be used as validation set.
'''
SEED = None  # Set to None for random seed.
BATCH_SIZE = 16  # Size of batch in training
NUM_EPOCHS = 15  # Number of epochs in training
RECORDING_STEP = 1000  # Number of records between two screen outputs.
# This helps to show a progress bar when training
TRAIN_FORMAT = 'data/training/images/satImage_{:03d}.png'
# Format of train images. Use '.format(number)' to get location of train image.
TRAIN_LABEL_FORMAT = 'data/training/groundtruth/satImage_{:03d}.png'
# Format of label images. Use '.format(number)' to get location of label image.
TEST_FORMAT = 'data/test_set_images/test_{}.png'
# Format of test images. Use '.format(number)' to get location of test image.
TEST_START = 1
'''
Test images are from 'test_(TEST_START).png'
to 'test_(TEST_START + TEST_SIZE - 1).png'.
'''
TEST_SIZE = 50
'''
TEST_START and TEST_SIZE can be set as any integers,
as long as TEST_START  > 0, TEST_SIZE > 0 and TEST_START + TEST_SIZE <= 51
The default TEST_START and TEST_SIZE will do predictions on all test images.
'''

TRAIN_MODEL = False
# if it's training(including doing predictions on remaining train images)
RESTORE_MODEL = True  # restore model from backup
TEST = True  # do predictions with test_set_images
TEST_SERVER = False  # if we do test on server.
'''
We assume that we have enough memory on server.
If TEST_SERVER is set False, memory that TensorFlow can use will be limited.
In our experience, it will be limited under 6G,
so we can do prediction on laptop.
'''
TEST_PATCH_SIZE = 19 * 19 * 64
'''
Size of patch that we use when we do predictions on test images.
If a computer has less memory, we can limit memory that the program can use
by decreasing TEST_PATCH_SIZE, for example to 19 * 19 * 4 or just 19 * 19.
However, small TEST_PATCH_SIZE will make the program slow.
'''
CONVERT_HSV = False  # if the images are converted to HSV domain.

IMG_PATCH_SIZE = 16  # Size of image patch.
# This is the size of one record of input data for this network.
IMG_STRIDE_SIZE = 2  # Stride that we use to get image patch.
# It's also size of label patch.
PADDING = int((IMG_PATCH_SIZE - IMG_STRIDE_SIZE) / 2)
# Calculate padding size that we need to add outside the whole image.

IMAGE_ROTATE = True
# if we rotate image patches randomly in the beginning of every epoch
IMAGE_HSV_RANDOM = False
# if we modify S and V channels of image patches randomly
# in the beginning of every epoch
BALANCE_DATA = False  # if we delete some train data to balance data
ADAM = False  # if we use Adam Optimizer
ADAM_INITIAL_RATE = 0.001  # initial learning rate of Adam Optimizer
MOMENTUM_INITIAL_RATE = 0.01  # initial learning rate of SGD Optimizer
MOMENTUM_MOMENTUM = 0.0  # momentum of MomentumOptimizer.
# We set this 0, so actually we use SGD Optimizer
DROPOUT = False  # if we do dropout

WEIGHT_STD = 0.01  # standard deviation of weight initial values.
# Weight initial values are generated with normal distribution


# Save key config to a file
# Then we can compare results of different experiences
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
    outf.write("CONVERT_HSV = {}\n".format(CONVERT_HSV))
    outf.write("ADAM = {}\n".format(ADAM))
    if ADAM:
        outf.write("INITIAL_RATE = {}\n".format(ADAM_INITIAL_RATE))
    else:
        outf.write("INITIAL_RATE = {}\n".format(MOMENTUM_INITIAL_RATE))
        outf.write("MOMENTUM = {}\n".format(MOMENTUM_MOMENTUM))
    outf.write("\n")
