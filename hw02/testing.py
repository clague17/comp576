from scipy import misc
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib as mp


# COMP 576 Assignnment 2 Task 1
# Grace Tan
# gzt1

# --------------------------------------------------
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

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    return W


def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    # maybe 0.0
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

    # IMPLEMENT YOUR CONV2D HERE
    h_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

    return h_conv


def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    h_max = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    return h_max


def variable_summaries(var):
    '''
    store summary of input var for tensorboard
    '''
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


ntrain = 1000  # per class
ntest = 100  # per class
nclass = 10  # number of classes
imsize = 28
nchannels = 1  # gray scale
batchsize = 128
result_dir = './myresults/'

Train = np.zeros((ntrain * nclass, imsize, imsize, nchannels))
Test = np.zeros((ntest * nclass, imsize, imsize, nchannels))
LTrain = np.zeros((ntrain * nclass, nclass))
LTest = np.zeros((ntest * nclass, nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = './CIFAR10/Train/%d/Image%05d.png' % (iclass, isample)
        im = misc.imread(path);  # 28 by 28
        im = im.astype(float) / 255
        itrain += 1
        Train[itrain, :, :, 0] = im
        LTrain[itrain, iclass] = 1  # 1-hot lable
    for isample in range(0, ntest):
        path = './CIFAR10/Test/%d/Image%05d.png' % (iclass, isample)
        im = misc.imread(path);  # 28 by 28
        im = im.astype(float) / 255
        itest += 1
        Test[itest, :, :, 0] = im
        LTest[itest, iclass] = 1  # 1-hot lable

sess = tf.InteractiveSession()

# tf variable for the data, remember shape is [None, width, height, numberOfChannels]
tf_data = tf.placeholder(tf.float32, shape=[None, imsize, imsize, nchannels])
# tf variable for labels
tf_labels = tf.placeholder(tf.float32, shape=[None, nclass])

# --------------------------------------------------
# model
# create your model

# for dropout
keep_prob = tf.placeholder(tf.float32)

# conv layer 5x5 kernel, 32 filter maps, relu
conv1W = weight_variable([5, 5, nchannels, 32])
conv1b = bias_variable([32])
conv1h = tf.nn.relu(conv2d(tf_data, conv1W) + conv1b)

with tf.name_scope('conv1h'):
    variable_summaries(conv1h)

# max pooling subsample 2
pool1h = max_pool_2x2(conv1h)
pool1h = tf.nn.dropout(pool1h, keep_prob)

with tf.name_scope('pool1h'):
    variable_summaries(pool1h)

# conv layer 5x5 kernel, 64 filter maps, relu
conv2W = weight_variable([5, 5, 32, 64])
conv2b = bias_variable([64])
conv2h = tf.nn.relu(conv2d(pool1h, conv2W) + conv2b)

with tf.name_scope('conv2h'):
    variable_summaries(convh)

# max pooling subsample 2
pool2h = max_pool_2x2(conv2h)
pool2h = tf.nn.dropout(pool2h, keep_prob)

with tf.name_scope('pool2h'):
    variable_summaries(pool2h)

# fully connected layer input 7*7*64 output 1024
fc1_W = weight_variable([7 * 7 * 64, 1024])
fc1_b = bias_variable([1024])
pool2h_flat = tf.reshape(pool2h, [-1, 7 * 7 * 64])
fc1_h = tf.matmul(pool2h_flat, fc1_W) + fc1_b

fc1_h = tf.nn.dropout(fc1_h, keep_prob)

# fully connected layer input 1024 output 10
# softmax in next step
fc2_W = weight_variable([1024, 10])
fc2_b = bias_variable([10])
y_conv = tf.matmul(fc1_h, fc2_W) + fc2_b

with tf.name_scope('y_conv(pred)'):
    variable_summaries(y_conv)

# --------------------------------------------------
# loss
# set up the loss, optimization, evaluation, and accuracy

learning_rate = 0.001
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=tf_labels))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(tf_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("Loss", cost)
tf.summary.scalar("Accuracy", accuracy)

summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(result_dir, sess.graph)

# --------------------------------------------------
# optimization

sess.run(tf.global_variables_initializer())

# setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_xs = np.zeros((batchsize, imsize, imsize, nchannels))
# setup as [batchsize, the how many classes]
batch_ys = np.zeros((batchsize, nclass))

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

nsamples = ntrain * nclass
training_iters = 10000
for i in range(training_iters):  # try a small iteration size once it works then continue
    perm = np.arange(nsamples)
    np.random.shuffle(perm)
    for j in range(batchsize):
        batch_xs[j, :, :, :] = Train[perm[j], :, :, :]
        batch_ys[j, :] = LTrain[perm[j], :]
    if i % 10 == 0:
        # calculate train accuracy and loss
        loss, acc = sess.run([cost, accuracy],
                             feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0})
        train_loss.append(loss)
        train_accuracy.append(acc)

        summary_str = sess.run(summary_op,
                               feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0})
        summary_writer.add_summary(summary_str, i)
        summary_writer.flush()

        # calculate test accuracy
        test_acc = accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0})
        test_accuracy.append(test_acc)
    if i % 100 == 0:
        # print stuff
        print("iteration: " + str(i) + ", loss= " + "{:.6f}".format(
            loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        print("test accuracy: ", "{:5f}".format(test_acc))

    # dropout only during training, optimize
    optimizer.run(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5})

# --------------------------------------------------
# test
print(
    "test accuracy %g" % accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))

# training and test accuracy curves
plt.plot(range(len(train_accuracy)), train_accuracy, 'b', label="Training Accuracy")
plt.plot(range(len(train_accuracy)), test_accuracy, "r", label="Testing Accuracy")
plt.title("Training and Test Accuracy")
plt.xlabel("Epochs (in thousands)")
plt.ylabel("Loss")
plt.legend()
plt.figure()

# train loss curve
plt.plot(range(len(train_loss)), train_loss, "b", label="Train Loss")
plt.title("Training Loss")
plt.xlabel("Epochs (in thousands)")
plt.ylabel("Loss")
plt.legend()
plt.figure()

# weights / filters of first convolutional layer
first_layer_weights = conv1W.eval()
for i in range(32):
    plt.subplot(4, 8, i + 1)
    plt.imshow(first_layer_weights[:, :, 0, i], cmap='gray')
    plt.title('Filter ' + str(i + 1))
    plt.axis('off')

plt.show()
sess.close()