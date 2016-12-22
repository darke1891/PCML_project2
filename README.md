# PCML_project2

## Introduction
This project is a CNN model for road segmentation. You can find more details in our report.

## Environment
This project is a pure Python project.

We use several libraries that you may need to install. The libraries are `TensorFlow`, `PIL`(Pillow), `numpy`, `matplotlib`.

However, as we generate many image patches, it needs a lot of memory to run the project. To train the model with the best parameters, you need to use about 40G memory. To do predictions on test data, we add a limitation and you need to use about 6G memory.

## Structure
All scripts are put 'src'.

`predict.py` is the TensorFlow model we use in this project. It's also the 'main' script.

`config.py` contains all parameters that you may need to train a new model or do predictions.

`basic_read.py` contains functions that read data from images. These functions are used by `test_read.py` and `train_read.py`.

`train_read.py` contains functions that generate data and labels for training.

`test_read.py` contains functions that generate data for predicting. It also read labels if we do predictions on train data.

`test_write.py` contains functions that save prediction results as images.

`mask_to_submission.py` contains functions that save prediction results as csv file.

## Do predictions
We have set parameter for testing, so you can run the prediction directly.

Please download test data from the competition and put them at folder 'src/data/', which means the first image is 'src/data/test_set_images/test_1.png'.

Then, run 'python predict.py' in 'src'.

For example, if test data are put at folder 'X'. Then run this script in this folder.

~~~~
cp -r X/test_set_images src/data/
cd src
python predict.py
~~~~

Then you can find prediction images in 'src/predictions_test', and csv file in 'src'.

### Hardware source
In our laptop, which has a 2-core intel i7-6600U processor and 16G memory, it takes about half an hour and about 6G memory to do predictions on all the 50 images.

If your machine has more than 30G memory, for example a cluster, you can set 'TEST_SERVER' in 'src/config.py' True, and it will be faster to do predictions.

If your machine has less memory, you can decrease 'TEST_PATCH_SIZE' in 'src/config.py'. You can find typical values in the file.