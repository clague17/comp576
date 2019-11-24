from scipy import misc
import os
import math
import torch
import numpy as np
import tensorflow as tf
from tqdm import tqdm # This import is for visualization and better printing :)
import random
import matplotlib.pyplot as plt
import matplotlib as mp

cwd = os.getcwd() # Setting the current working dir for output.

#####################################################
"""
COMP 576 HW 2
Luis Clague
Fall 2019
"""
#####################################################

# setup

def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    return W

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''
    b = tf.Variable(tf.constant(0.0, shape=shape))
    return b

def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''
    h_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return h_conv

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''
    h_max = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return h_max

# helpers
def var_log(name, vector):
    """
    Helper for logging to tensorboard for each variable
    :param name: The variable name
    :param vector: The actual variable
    :return: void
    """
    with tf.name_scope(name + "_SUMMARY"):
        mean, variance = tf.nn.moments(vector, axes=list(range(tf.rank(vector).eval())))
        tf.summary.scalar("mean", mean)
        tf.summary.scalar("variance", variance)
        tf.summary.scalar("std", tf.sqrt(variance))
        tf.summary.scalar("max", tf.reduce_max(vector))
        tf.summary.scalar("min", tf.reduce_min(vector))
        tf.summary.histogram('Hist', vector)

class checkpoint:
    def __init__(self, save_to, session):
        self.cur_max = -float('inf')
        self.saver = tf.train.Saver()
        self.save_to = save_to

    def poll(self, new_val):
        if new_val >= self.cur_max:
            print("[BETTER MODEl ACCURACY]: ", new_val, "- saving model.")
            self.saver.save(sess, os.path.join(cwd, self.save_to))
            self.cur_max = new_val

ntrain = 1000 # per class
ntest = 100 # per class
nclass = 10 # number of classes
imsize = 28
nchannels = 1
batchsize = 128

output_dir = "cifar_results"

Train = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
Test = np.zeros((ntest*nclass,imsize,imsize,nchannels))
LTrain = np.zeros((ntrain*nclass,nclass))
LTest = np.zeros((ntest*nclass,nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = './CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path) # 28 by 28
        im = im.astype(float)/255
        itrain += 1
        Train[itrain,:,:,0] = im
        LTrain[itrain,iclass] = 1 # 1-hot label
    for isample in range(0, ntest):
        path = './CIFAR10/Test/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itest += 1
        Test[itest,:,:,0] = im
        LTest[itest,iclass] = 1 # 1-hot label

sess = tf.InteractiveSession()

tf_data = tf.placeholder(tf.float32, shape=[None, imsize, imsize, nchannels])#tf variable for the data, remember shape is [None, width, height, numberOfChannels]
tf_labels = tf.placeholder(tf.float32, shape=[None, nclass])#tf variable for labels

# model
#create your model
# conv_prob is used for the dropout
conv_prob =  tf.placeholder(tf.float32)

# First layer: 5x5 / 32 filters, ReLu activation
W1 = weight_variable([5, 5, 1, 32])
b1 = bias_variable([32])
h1 = tf.nn.relu(conv2d(tf_data, W1) + b1)

var_log("W1", W1)
var_log("b1", b1)
var_log("h1", h1)

h_pool1 = max_pool_2x2(h1)
h_pool1_dropout = tf.nn.dropout(h_pool1, conv_prob)
var_log("h_pool1", h_pool1)

# Second layer: 5x5 / 64, ReLu

W2 = weight_variable([5, 5, 32, 64])
b2 = bias_variable([64])
h2 = tf.nn.relu(conv2d(h_pool1_dropout, W2) + b2)
h_pool2 = max_pool_2x2(h2)
h_pool2_dropoff = tf.nn.dropout(h_pool2, conv_prob)

var_log("W2", W2)
var_log("b2", b2)
var_log("h2", h2)
var_log("h_pool2", h_pool2)

# fully connected layer
# Layer 3: 7x7x64 in / 1024 out
flat_shape = 7 * 7 * 64
fc1_W = weight_variable([flat_shape, 1024])
fc1_b = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2_dropoff, [-1, flat_shape])
fc1_h = tf.matmul(h_pool2_flat, fc1_W) + fc1_b

fc1_h_dropout = tf.nn.dropout(fc1_h, conv_prob)

# Fully Connected layer
#Layer 4: 1024 in / 10 out
fc2_W = weight_variable([1024, 10])
fc2_b = bias_variable([10])
y_conv = tf.matmul(fc1_h_dropout, fc2_W) + fc2_b

conv_out = h_pool2_dropoff

var_log("fc2_W", fc2_W)
var_log("fc2_b", fc2_b)
var_log("y_conv", y_conv)

# --------------------------------------------------
# loss
# set up the loss, optimization, evaluation, and accuracy
# setup training
# --------------------------------------------------
learn_rate = .001 # MODIFIED this learning rate as a variable for accuracy.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_labels, logits=y_conv))
optimizer = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(tf_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("Loss", cross_entropy)
tf.summary.scalar("Accuracy", accuracy)

# Set up the summary writer
result_dir = os.path.join(cwd, output_dir)
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(result_dir, sess.graph)


# --------------------------------------------------
# optimization
sess.run(tf.global_variables_initializer())

# Create model checkpointer
checkpointer = checkpoint("cifar10-best", sess)

batch_xs = np.empty((batchsize, imsize, imsize, nchannels))
batch_ys = np.zeros((batchsize, nclass))
nsamples = ntrain * nclass
# batch indices
periods = 50

for i in range(12000):
    perm = np.arange(nsamples)
    np.random.shuffle(perm)
    for j in range(batchsize):
        batch_ys[j, :] = LTrain[perm[j], :]
        batch_xs[j, :, :, :] = Train[perm[j], :, :, :]
    if i % 10 == 0:
        train_accuracy, train_loss = sess.run([accuracy, cross_entropy], feed_dict={
            tf_data: batch_xs, tf_labels: batch_ys, conv_prob: 1.0})
        summary_str = sess.run(summary_op, feed_dict={
            tf_data: batch_xs, tf_labels: batch_ys, conv_prob: 1.0})
        summary_writer.add_summary(summary_str, i)
        summary_writer.flush()
        test_acc = accuracy.eval(
            feed_dict={tf_data: Test, tf_labels: LTest, conv_prob: 1.0})
        # Collect the summary statistics on test data
        summary_str = sess.run(summary_op, feed_dict={
            tf_data: Test, tf_labels: LTest, conv_prob: 1.0})
        summary_writer.add_summary(summary_str, i)
        summary_writer.flush()
    if i % 100 == 0:
        # print stuff
        # print("iteration: " + str(i) + ", loss= " + "{:.6f}".format(
        #     train_loss) + ", Training Accuracy= " + "{:.5f}".format(train_accuracy))
        # print("test accuracy: ", "{:5f}".format(test_acc))
        checkpointer.poll(test_acc)
    # dropout only during training
    optimizer.run(feed_dict={
        tf_data: batch_xs, tf_labels: batch_ys, conv_prob: 0.5})

# test
print("test accuracy %g"%accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, conv_prob: 1.0}))

# first layer weights / filters
layer_weights = W1.eval()

for i in range(32):
    plt.subplot(4, 8, i + 1)
    plt.imshow(layer_weights[:, :, 0, i], cmap="gray")
    plt.title("Filter " + str(i + 1))
    plt.axis("off")
plt.show()

second_layer = W2.eval()

for i in range(32):
    plt.subplot(4, 8, i + 1)
    plt.imshow(second_layer[:, :, 0, i], cmap="gray")
    plt.title("Filter " + str(i + 1))
    plt.axis("off")


plt.show()
sess.close()