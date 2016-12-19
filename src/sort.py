"""
Baseline for machine learning project on road segmentation.
This simple baseline consits of a CNN with two convolutional+pooling layers with a soft-max loss

Credits: Aurelien Lucchi, ETH Zürich
"""


import os
import sys
import csv
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

from PIL import Image

from config import *
from basic_read import read_images, extract_labels
from train_read import extract_train
from test_read import extract_test_labels, extract_test_data
from test_write import save_image
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

tf.app.flags.DEFINE_string('train_dir', '/tmp/mnist-baoge',   # don't use default folder
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS


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




def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
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
    print (str(max_labels) + ' ' + str(max_predictions))


# Make an image summary for 4d tensor image with index idx
def get_image_summary(img, idx = 0):
    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    img_w = img.get_shape().as_list()[1]
    img_h = img.get_shape().as_list()[2]
    min_value = tf.reduce_min(V)
    V = V - min_value
    max_value = tf.reduce_max(V)
    V = V / (max_value*PIXEL_DEPTH)
    V = tf.reshape(V, (img_w, img_h, 1))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, (-1, img_w, img_h, 1))
    return V


# Make an image summary for 3d tensor image with index idx
def get_image_summary_3d(img):
    V = tf.slice(img, (0, 0, 0), (1, -1, -1))
    img_w = img.get_shape().as_list()[1]
    img_h = img.get_shape().as_list()[2]
    V = tf.reshape(V, (img_w, img_h, 1))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, (-1, img_w, img_h, 1))
    return V


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
    # if train:
    #     reshape = tf.nn.dropout(reshape, 0.5, seed=SEED)
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, kwargs['fc1_weights']) + kwargs['fc1_biases'])
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    # if train:
    #     hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    out = tf.matmul(hidden, kwargs['fc2_weights']) + kwargs['fc2_biases']

    if train:
        summary_id = '_0'
        s_data = get_image_summary(data)
        filter_summary0 = tf.summary.image('summary_data' + summary_id, s_data)
        s_conv = get_image_summary(conv)
        filter_summary2 = tf.summary.image('summary_conv' + summary_id, s_conv)
        s_pool = get_image_summary(pool)
        filter_summary3 = tf.summary.image('summary_pool' + summary_id, s_pool)
        s_conv2 = get_image_summary(conv2)
        filter_summary4 = tf.summary.image('summary_conv2' + summary_id, s_conv2)
        s_pool2 = get_image_summary(pool2)
        filter_summary5 = tf.summary.image('summary_pool2' + summary_id, s_pool2)

    return out


def train(s, saver, all_params, outf=None):

    # Extract it into np arrays.
    train_data, train_labels = extract_train()

    num_epochs = NUM_EPOCHS

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 += 1
        else:
            c1 += 1
    print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    if BALANCE_DATA:
        print ('Balancing training data...')
        min_c = min(c0, c1)
        idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
        idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
        new_indices = idx0[0:min_c] + idx1[0:min_c]
        print (len(new_indices))
        train_data = train_data[new_indices, :, :, :]
        train_labels = train_labels[new_indices]


        c0 = 0
        c1 = 0
        for i in range(len(train_labels)):
            if train_labels[i][0] == 1:
                c0 += 1
            else:
                c1 += 1
        print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    train_size = train_labels.shape[0]
    print('Training size is {}'.format(train_size))

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(
            tf.float32,
            shape=(BATCH_SIZE, NUM_LABELS))
    train_all_data_place = tf.placeholder(tf.float32, shape=train_data.shape)
    train_all_data_node = tf.Variable(train_all_data_place)

    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, True, all_params) # BATCH_SIZE*NUM_LABELS
    # print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits, train_labels_node))
    tf.summary.scalar('loss', loss)

    all_params_names = list(all_params.keys())
    all_params_nodes = list(all_params.values())
    all_grads_node = tf.gradients(loss, all_params_nodes)
    all_grad_norms_node = []
    for i in range(0, len(all_grads_node)):
        norm_grad_i = tf.global_norm([all_grads_node[i]])
        all_grad_norms_node.append(norm_grad_i)
        tf.summary.scalar(all_params_names[i], norm_grad_i)
    
    # L2 regularization for the fully connected parameters.
    regularizers = (
            tf.nn.l2_loss(all_params['fc1_weights']) +
            tf.nn.l2_loss(all_params['fc1_biases']) +
            tf.nn.l2_loss(all_params['fc2_weights']) + 
            tf.nn.l2_loss(all_params['fc2_biases']))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)

    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        MOMENTUM_INITIAL_RATE,  # Base learning rate.
        batch * BATCH_SIZE,     # Current index into the dataset.
        train_size,             # Decay step.
        0.95,                   # Decay rate.
        staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    
    # Use simple momentum for the optimization.
    if ADAM:
        optimizer = tf.train.AdamOptimizer(ADAM_INITIAL_RATE).minimize(loss, global_step=batch)
    else:
        optimizer = tf.train.MomentumOptimizer(learning_rate, MOMENTUM_MOMENTUM).minimize(loss, global_step=batch)

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # We'll compute them only once in a while by calling their {eval()} method.
    train_all_prediction = tf.nn.softmax(model(train_all_data_node, False, all_params))

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                            graph_def=s.graph_def)
    # If variables are not restored, 
    # run all the initializers to prepare the trainable parameters.
    if RESTORE_MODEL:
        all_vars = tf.all_variables()
        uninit_vars = [var for var in all_vars if not tf.is_variable_initialized(var).eval()]
        print('The following are uninitialized variables. Start initing them ')
        print([var.name for var in uninit_vars])
        tf.initialize_variables(uninit_vars).run(
                feed_dict={train_all_data_place: train_data})
    else:
        tf.initialize_all_variables().run(feed_dict={train_all_data_place: train_data})

    print ('Initialized!')
    # Loop through training steps.
    print ('Total number of iterations = ' + str(int(num_epochs * train_size / BATCH_SIZE)))

    if outf is not None:
        outf.write("TRAIN_SIZE = {}\n".format(TRAIN_SIZE))
        outf.write("NUM_EPOCHS = {}\n".format(NUM_EPOCHS))
        outf.write("BATCH_SIZE = {}\n".format(BATCH_SIZE))
        outf.write("IMG_PATCH_SIZE = {}\n".format(IMG_PATCH_SIZE))
        outf.write("IMG_STRIDE_SIZE = {}\n".format(IMG_STRIDE_SIZE))
        outf.write("IMAGE_ROTATE = {}\n".format(IMAGE_ROTATE))
        outf.write("IMAGE_HSV_RANDOM = {}\n".format(IMAGE_HSV_RANDOM))
        outf.write("BALANCE_DATA = {}\n".format(BALANCE_DATA))
        outf.write("ADAM = {}\n".format(ADAM))
        if ADAM:
            outf.write("ADAM_INITIAL_RATE = {}\n".format(ADAM_INITIAL_RATE))
        else:
            outf.write("MOMENTUM_INITIAL_RATE = {}\n".format(MOMENTUM_INITIAL_RATE))
            outf.write("MOMENTUM_MOMENTUM = {}\n".format(MOMENTUM_MOMENTUM))
        outf.write("\n")

    training_indices = range(train_size)

    for iepoch in range(num_epochs):

        # Permute training indices
        perm_indices = np.random.permutation(training_indices)

        for step in range (int(train_size / BATCH_SIZE)):

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

                summary_str, _, l, lr, predictions = s.run(
                    [summary_op, optimizer, loss, learning_rate, train_prediction],
                    feed_dict=feed_dict)
                #summary_str = s.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

                # print_predictions(predictions, batch_labels)

                print ('Epoch {:2f}, {}/{}' .format(float(step) * BATCH_SIZE / train_size, iepoch, num_epochs))
                print ('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print ('Minibatch error: %.1f%%' % error_rate(predictions,
                                                             batch_labels))

                sys.stdout.flush()
            else:
                # Run the graph and fetch some of the nodes.
                _, l, lr, predictions = s.run(
                    [optimizer, loss, learning_rate, train_prediction],
                    feed_dict=feed_dict)

        # Save the variables to disk.
        # save the last one
        save_path = saver.save(s, FLAGS.train_dir + "/model.ckpt", write_meta_graph=False)
        print("Model saved in file: %s" % save_path)
        
        if outf is not None:
            outf.write("Epoch {}\n".format(iepoch))
        # do predictions on remaining training data
        remaining_train_size = 100 - TRAIN_SIZE
        if remaining_train_size != 0:
            remaining_start = TRAIN_SIZE + 1
            print("""Cross validation on train data.\nRemaining {} train images will 
                    be used for testing""".format(remaining_train_size))

            if outf is not None:
                outf.write("Validation Set: ")
            test(s, all_params, TRAIN_FORMAT, remaining_start, remaining_train_size, outf)

        print ("Prediction on train data")
        if outf is not None:
            outf.write("Train Set: ")
        test(s, all_params, TRAIN_FORMAT, TRAIN_START, TRAIN_SIZE, outf)

def test(s, all_params, data_format, index_start, size, outf=None):
    images = extract_test_data(data_format, index_start, size)
    is_cv = 'train' in data_format
    prediction_dir = "predictions_test/"
    if is_cv:
        prediction_dir = 'predictions_train/'
        labels = extract_labels(TRAIN_LABEL_FORMAT, index_start, size, False)
    
    if not os.path.isdir(prediction_dir):
        os.mkdir(prediction_dir)

    output_predictions = np.zeros((0, NUM_LABELS))
    for index, image_data in enumerate(images):
        img = image_data[0]
        img_patches = image_data[1]
        # predicting
        data_node = tf.constant(img_patches)
        output = tf.nn.softmax(model(data_node, False, all_params))
        prediction = s.run(output)
        output_predictions = np.concatenate((output_predictions, prediction))

        save_image(img[PADDING:(img.shape[0]-PADDING),PADDING:(img.shape[1]-PADDING),:], prediction, prediction_dir, index + index_start)

    if is_cv:
        error_rate_str = 'Error rate: {}'.format(
            error_rate(output_predictions, labels))
        print(error_rate_str)
        if outf is not None:
            outf.write(error_rate_str + "\n")


def main(args=None):
    out_name = get_outf_name()
    with open(out_name, 'w') as outf:
        # The variables below hold all the trainable weights. They are passed an
        # initial value which will be assigned when when we call:
        # {tf.initialize_all_variables().run()}
        conv1_weights = tf.Variable(
            tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                                stddev=0.1,
                                seed=SEED),
            name='conv1_weights')
        conv1_biases = tf.Variable(tf.zeros([32]), name='conv1_biases')
        conv2_weights = tf.Variable(
            tf.truncated_normal([5, 5, 32, 64],
                                stddev=0.1,
                                seed=SEED),
            name='conv2_weights')
        conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]), name='conv2_biases')
        fc1_weights = tf.Variable(  # fully connected, depth 1024.
            tf.truncated_normal([int(IMG_PATCH_SIZE / 4 * IMG_PATCH_SIZE / 4 * 64), 1024],
                                stddev=0.1,
                                seed=SEED),
            name='fc1_weights')
        fc1_biases = tf.Variable(tf.constant(0.1, shape=[1024]), name='fc1_biases')
        fc2_weights = tf.Variable(
            tf.truncated_normal([1024, NUM_LABELS],
                                stddev=0.1,
                                seed=SEED),
            name='fc2_weigths')
        fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]), name='fc2_biases')


        all_params_node = [conv1_weights, conv1_biases, conv2_weights, conv2_biases, fc1_weights, fc1_biases, fc2_weights,
                           fc2_biases]
        all_params_names = ['conv1_weights', 'conv1_biases', 'conv2_weights', 'conv2_biases', 'fc1_weights', 'fc1_biases',
                            'fc2_weights', 'fc2_biases']
        all_params = dict(zip(all_params_names, all_params_node))

        saver = tf.train.Saver() 

        with tf.Session(
                config=tf.ConfigProto(
                    intra_op_parallelism_threads=10,
                    inter_op_parallelism_threads=10,
                    use_per_session_threads=True)
                ) as s:
            # read saved model
            if RESTORE_MODEL:
                saver.restore(s, FLAGS.train_dir + "/model.ckpt")
                print('Model Restored! The following variables are stored: ')
                print_tensors_in_checkpoint_file(FLAGS.train_dir + '/model.ckpt', False)
            # train the saved model or a new model
            # including cross validation on 100-TRAIN_SIZE images
            if TRAIN_MODEL:
                train(s, saver, all_params, outf)

            # predicting on test data
            if TEST:
                test(s, all_params, TEST_FORMAT, TEST_START, TEST_SIZE, outf)


if __name__ == '__main__':
    tf.app.run()
