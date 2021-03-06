"""
Our CNN model.
There are two CNN layers and one fully connected layer.
In our test, all parameters that need to be adjusted are put in 'config.py',
so usually you don't need to modify this file, unless you want to modify
struct of the network.

This model is based on baseline model for machine learning
project on road segmentation.
"""


import os
import sys
import csv
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

from datetime import datetime

from PIL import Image
from config import *
from train_read import extract_train
from test_read import extract_test_labels, extract_test_data
from test_write import save_image
from mask_to_submission import masks_to_submission

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

tf.app.flags.DEFINE_string('train_dir', MODEL_DIR,
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS


# generate a file name that we can save information in this train
def get_outf_name():
    out_dir = "validation/"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    index = 0
    while (True):
        out_file_name = out_dir + "test" + str(index)
        if os.path.exists(out_file_name):
            index += 1
        else:
            break
    return out_file_name


# Return the error rate based on predictions and labels.
# We use error rate but not F1 score.
def error_rate(predictions, labels):
    error_rate = 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
        predictions.shape[0])
    return error_rate


# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = np.argmax(labels, 1)
    max_predictions = np.argmax(predictions, 1)
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        for l, p in zip(max_labels, max_predictions):
            writer.writerow(l, p)


# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = np.argmax(labels, 1)
    max_predictions = np.argmax(predictions, 1)
    print(str(max_labels) + ' ' + str(max_predictions))


# We will replicate the model structure for the training subgraph, as well
# as the evaluation subgraphs, while sharing the trainable parameters.
def model(data, train, kwargs):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        kwargs['conv1_weights'],
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, kwargs['conv1_biases']))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

    conv2 = tf.nn.conv2d(pool,
                         kwargs['conv2_weights'],
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, kwargs['conv2_biases']))
    pool2 = tf.nn.max_pool(relu2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')

    # Uncomment these lines to check the size of each layer
    # print 'data ' + str(data.get_shape())
    # print 'conv ' + str(conv.get_shape())
    # print 'relu ' + str(relu.get_shape())
    # print 'pool ' + str(pool.get_shape())
    # print 'pool2 ' + str(pool2.get_shape())

    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool2.get_shape().as_list()
    reshape = tf.reshape(
        pool2,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train and DROPOUT:
        reshape = tf.nn.dropout(reshape, 0.5, seed=SEED)
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, kwargs['fc1_weights']) +
                        kwargs['fc1_biases'])
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train and DROPOUT:
        hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    out = tf.matmul(hidden, kwargs['fc2_weights']) + kwargs['fc2_biases']

    return out


# Train the model and test on both train set and validation set every epoch
def train(s, saver, all_params, outf=None):

    # Extract it into np arrays.
    train_data, train_labels = extract_train()

    num_epochs = NUM_EPOCHS

    c0 = 0  # number of data records whose label is 0
    c1 = 0  # number of data records whose label is 1
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 += 1
        else:
            c1 += 1
    print('Number of data points per class: c0 = ' +
          str(c0) + ' c1 = ' + str(c1))

    if BALANCE_DATA:
        print('Balancing training data...')
        min_c = min(c0, c1)
        idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
        idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
        np.random.shuffle(idx0)
        np.random.shuffle(idx1)
        new_indices = idx0[0:min_c] + idx1[0:min_c]
        print(len(new_indices))
        train_data = train_data[new_indices, :, :, :]
        train_labels = train_labels[new_indices]

        # Calculate number of data records again
        c0 = 0
        c1 = 0
        for i in range(len(train_labels)):
            if train_labels[i][0] == 1:
                c0 += 1
            else:
                c1 += 1
        print('Number of data points per class: c0 = ' +
              str(c0) + ' c1 = ' + str(c1))

    train_size = train_labels.shape[0]

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(tf.float32,
                                     shape=(BATCH_SIZE,
                                            IMG_PATCH_SIZE,
                                            IMG_PATCH_SIZE,
                                            NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE,
                                              NUM_LABELS))
    train_all_data_place = tf.placeholder(tf.float32, shape=train_data.shape)
    print('train_data shape {}'.format(train_data.shape))
    train_all_data_node = tf.Variable(train_all_data_place)

    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, True, all_params)
    # BATCH_SIZE*NUM_LABELS

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits, train_labels_node))

    if not DROPOUT:
        # L2 regularization for the fully connected parameters.
        # Dropout can replace this regularization
        regularizers = (tf.nn.l2_loss(all_params['fc1_weights']) +
                        tf.nn.l2_loss(all_params['fc1_biases']) +
                        tf.nn.l2_loss(all_params['fc2_weights']) +
                        tf.nn.l2_loss(all_params['fc2_biases']))
        # Add the regularization term to the loss.
        loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)

    # Use simple momentum for the optimization.
    if ADAM:
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        learning_rate = tf.train.exponential_decay(
            ADAM_INITIAL_RATE,  # Base learning rate.
            batch * BATCH_SIZE,     # Current index into the dataset.
            train_size,             # Decay step.
            0.95,                   # Decay rate.
            staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch)
    else:
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        learning_rate = tf.train.exponential_decay(
            MOMENTUM_INITIAL_RATE,  # Base learning rate.
            batch * BATCH_SIZE,     # Current index into the dataset.
            train_size,             # Decay step.
            0.95,                   # Decay rate.
            staircase=True)
        optimizer = tf.train.MomentumOptimizer(learning_rate, MOMENTUM_MOMENTUM).minimize(loss, global_step=batch)

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # We'll compute them only once in a while by calling their {eval()} method.
    train_all_prediction = tf.nn.softmax(model(train_all_data_node,
                                               False, all_params))

    # If variables are not restored,
    # run all the initializers to prepare the trainable parameters.
    if RESTORE_MODEL:
        # initialize uninitialized variables if we can restore the model
        all_vars = tf.all_variables()
        uninit_vars = [var for var in all_vars if not tf.is_variable_initialized(var).eval()]
        print('The following are uninitialized variables. Start initing them ')
        print([var.name for var in uninit_vars])
        tf.variables_initializer(uninit_vars).run(feed_dict={train_all_data_place: train_data})
    else:
        # otherwise, initialize all variables
        tf.initialize_all_variables().run(feed_dict={train_all_data_place: train_data})

    print('Initialized!')
    # Loop through training steps.
    print('Total number of iterations = ' +
          str(int(num_epochs * train_size / BATCH_SIZE)))

    # save configure to file
    if outf is not None:
        print_config(outf)

    training_indices = range(train_size)

    for iepoch in range(num_epochs):
        print(datetime.now())

        if IMAGE_ROTATE:
            # for each image, give it a random integer k between 0 and 3
            # then rotate it k * 90 degree
            rotate_list = np.random.randint(4, size=train_data.shape[0])
            for index in range(0, train_data.shape[0]):
                train_data[index, :] = np.rot90(train_data[index, :],
                                                rotate_list[index])
        if IMAGE_HSV_RANDOM:
            # for each image, give it two random number between 0.9 and 1.1
            # then multiply S and V channels by this factor.
            hsv_random = np.random.uniform(0.9, 1.1, (train_data.shape[0], 2))
            for index in range(0, train_data.shape[0]):
                train_data[index, :, :, 1] *= hsv_random[index, 0]
                train_data[index, :, :, 2] *= hsv_random[index, 1]

        # Permute training indices
        perm_indices = np.random.permutation(training_indices)

        for step in range(int(train_size / BATCH_SIZE)):
            # get indices of this batch
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]

            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            batch_data = train_data[batch_indices, :, :, :]
            batch_labels = train_labels[batch_indices]
            # This dictionary maps the batch data (as a np array) to the
            # node in the graph is should be fed to.
            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}

            if step % RECORDING_STEP == 0:
                # print progress bar
                # then print '\r' and return to beginning of the line
                # this makes dynamic progress bar
                print('Epoch {:2f}, {}/{}'
                      .format(float(step) * BATCH_SIZE / train_size,
                              iepoch, num_epochs),
                      end='\r')

            _, l, lr, predictions = s.run(
                [optimizer, loss, learning_rate, train_prediction],
                feed_dict=feed_dict)
        # print new line after one epoch finishes
        print('')

        # Save the variables to disk.
        # save the last one
        save_path = saver.save(s, "{}/model{}.ckpt".format(FLAGS.train_dir,
                                                           iepoch),
                               write_meta_graph=False)
        print("Model saved in file: {}".format(save_path))

        if outf is not None:
            outf.write("Epoch {}\n".format(iepoch))
        # do predictions on remaining training data
        remaining_train_size = 100 - TRAIN_SIZE
        if remaining_train_size != 0:
            remaining_start = TRAIN_SIZE + 1
            print(('Cross validation on train data.\n'
                   'Remaining {} train images will be used for testing')
                  .format(remaining_train_size))
            if outf is not None:
                outf.write("Validation Set: ")
            test(s, all_params, TRAIN_FORMAT, remaining_start, remaining_train_size, outf)

        # do predictions on train set
        print("Prediction on train set")
        if outf is not None:
            outf.write("Train Set: ")
        test(s, all_params, TRAIN_FORMAT, TRAIN_START, TRAIN_SIZE, outf)
        print('-' * 30)

        if IMAGE_ROTATE:
            # restore data
            for index in range(0, train_data.shape[0]):
                train_data[index, :] = np.rot90(train_data[index, :],
                                                4 - rotate_list[index])
        if IMAGE_HSV_RANDOM:
            # restore data
            for index in range(0, train_data.shape[0]):
                train_data[index, :, :, 1] /= hsv_random[index, 0]
                train_data[index, :, :, 2] /= hsv_random[index, 1]


# do prediction on a data set
# use data_format to choose folder that contains data
# use index_start and size to choose image
def test(s, all_params, data_format, index_start, size, outf=None):
    images = extract_test_data(data_format, index_start, 1)
    # if we can find 'train' in the folder, then it's train data
    is_cv = 'train' in data_format
    prediction_dir = "predictions_test/"
    if is_cv:
        # we know true labels of train data
        # so we can calcualte error rate later
        prediction_dir = 'predictions_train/'
        labels = extract_test_labels(index_start, size)

    if not os.path.isdir(prediction_dir):
        os.mkdir(prediction_dir)

    output_predictions = np.zeros((0, NUM_LABELS))

    if is_cv or TEST_SERVER:
        # use this placeholder for every images
        data_place = tf.placeholder(tf.float32, shape=images[0][1].shape)
    else:
        # if we do predictions on laptop, we don't
        # have enough memory for large data
        # so we divide data into several parts and
        # send less data to TensorFlow ever time
        data_place = tf.placeholder(tf.float32,
                                    shape=images[0][1][0:TEST_PATCH_SIZE, :].shape)

    data_node = tf.Variable(data_place)

    for index in range(index_start, index_start + size):
        if not is_cv:
            print("process picture " + str(index))
        # read images one by one
        # otherwise it will need more than 10G memory
        images = extract_test_data(data_format, index, 1)
        image_data = images[0]
        img = image_data[0]
        img_patches = image_data[1]

        # predicting
        if is_cv or TEST_SERVER:
            tf.variables_initializer([data_node]).run(feed_dict={data_place: img_patches})
            output = tf.nn.softmax(model(data_node, False, all_params))
            prediction = s.run(output)
        else:
            prediction = np.zeros((0, NUM_LABELS))
            for start in range(0, img_patches.shape[0], TEST_PATCH_SIZE):
                end = start + TEST_PATCH_SIZE
                img_patches_patch = img_patches[start:end, :]
                tf.variables_initializer([data_node]).run(feed_dict={data_place: img_patches_patch})
                output = tf.nn.softmax(model(data_node, False, all_params))
                prediction_patch = s.run(output)
                prediction = np.concatenate((prediction, prediction_patch))

        output_predictions = np.concatenate((output_predictions, prediction))
        save_image(img[PADDING:(img.shape[0] - PADDING),
                       PADDING:(img.shape[1] - PADDING), :],
                   prediction, prediction_dir, index)

    if is_cv:
        # When training, calculate error rate and save it
        error_rate_str = 'Error rate: {}'.format(
            error_rate(output_predictions, labels))
        print(error_rate_str)
        if outf is not None:
            outf.write(error_rate_str + "\n")


def main(args=None):
    if not os.path.isdir(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=WEIGHT_STD,
                            seed=SEED),
        name='conv1_weights')
    conv1_biases = tf.Variable(tf.zeros([32]), name='conv1_biases')
    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64],
                            stddev=WEIGHT_STD,
                            seed=SEED),
        name='conv2_weights')
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]),
                               name='conv2_biases')
    fc1_weights = tf.Variable(  # fully connected, depth 1024.
        tf.truncated_normal([int(IMG_PATCH_SIZE / 4 *
                                 IMG_PATCH_SIZE / 4 * 64), 1024],
                            stddev=WEIGHT_STD,
                            seed=SEED),
        name='fc1_weights')
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[1024]), name='fc1_biases')
    fc2_weights = tf.Variable(
        tf.truncated_normal([1024, NUM_LABELS],
                            stddev=WEIGHT_STD,
                            seed=SEED),
        name='fc2_weigths')
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]),
                             name='fc2_biases')

    all_params_node = [conv1_weights, conv1_biases, conv2_weights,
                       conv2_biases, fc1_weights, fc1_biases,
                       fc2_weights, fc2_biases]
    all_params_names = ['conv1_weights', 'conv1_biases', 'conv2_weights',
                        'conv2_biases', 'fc1_weights', 'fc1_biases',
                        'fc2_weights', 'fc2_biases']
    all_params = dict(zip(all_params_names, all_params_node))

    saver = tf.train.Saver(max_to_keep=20)

    with tf.Session(config=tf
                    .ConfigProto(intra_op_parallelism_threads=NUM_THREADS,
                                 inter_op_parallelism_threads=NUM_THREADS,
                                 use_per_session_threads=True)
                    ) as s:
        # read saved model
        if RESTORE_MODEL:
            saver.restore(s, FLAGS.train_dir + "/model.ckpt")
            print('Model Restored! The following variables are stored: ')
            print_tensors_in_checkpoint_file(FLAGS.train_dir + '/model.ckpt',
                                             False)
        # train the saved model or a new model
        # including cross validation on 100-TRAIN_SIZE images
        if TRAIN_MODEL:
            out_name = get_outf_name()
            with open(out_name, 'w') as outf:
                train(s, saver, all_params, outf)

        # predicting on test data
        if TEST:
            test(s, all_params, TEST_FORMAT, TEST_START, TEST_SIZE)
            submission_filename = 'submission.csv'
            image_filenames = []
            for i in range(1, 51):
                image_filename = 'predictions_test/prediction_{}.png'.format(i)
                image_filenames.append(image_filename)
            masks_to_submission(submission_filename, *image_filenames)


if __name__ == '__main__':
    tf.app.run()
