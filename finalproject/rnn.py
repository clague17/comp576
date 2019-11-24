import tensorflow as tf
import tensorflow_datasets as tfds
# from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
# from torch.utils.tensorboard import SummaryWriter


# from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

# mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

########## groove starter code ###################

# tfds works in both Eager and Graph modes
tf.enable_eager_execution()

# Load the full GMD with MIDI only (no audio) as a tf.data.Dataset
dataset = tfds.load(
    name="groove/2bar-midionly",
    split=tfds.Split.TRAIN,
    try_gcs=True)

print("loaded dataset")

# Build your input pipeline
dataset = dataset.shuffle(1024).batch(32).prefetch(
    tf.data.experimental.AUTOTUNE)
i = 0
for features in dataset.take(1):
    # Access the features you are interested in
    midi, genre = features["midi"], features["style"]["primary"]
    if i < 100:
      print(midi, genre)




##################################################

learningRate = .003
trainingIters = 100000
batchSize = 128
displayStep = 10

nInput = 28  # we want the input to take the 28 pixels
nSteps = 28  # every 28
nHidden = 256  # number of neurons for the RNN
nClasses = 10  # this is MNIST so you know

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = {
    'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
    'out': tf.Variable(tf.random_normal([nClasses]))
}


def RNN(x, weights, biases, gru=False):
    # configuring so you can get it as needed for the 28 pixels
    x = tf.unstack(x, nSteps, 1)

    # find which lstm to use in the documentation
    lstmCell = rnn.GRUCell(nHidden) if gru else rnn.BasicLSTMCell(nHidden)

    # for the rnn where to get the output and hidden state
    outputs, states = tf.nn.static_rnn(lstmCell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

"""
pred = RNN(x, weights, biases, gru=False)

# optimization
# create the cost, optimization, evaluation, and accuracy
# for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(
    learning_rate=learningRate).minimize(cost)

correctPred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

init = tf.global_variables_initializer()

testData = mnist.test.images.reshape((-1, nSteps, nInput))
testLabel = mnist.test.labels
"""
"""
with tf.Session() as sess:
    sess.run(init)

    writer = SummaryWriter()
    step = 1
    counter = 1

    while step * batchSize < trainingIters:
        # mnist has a way to get the next batch
        batchX, batchY = mnist.train.next_batch(batchSize)
        batchX = batchX.reshape((batchSize, nSteps, nInput))

        sess.run(optimizer, feed_dict={x: batchX, y: batchY})

        if step % displayStep == 0:
            counter += 1
            # Running the training
            acc = accuracy.eval(feed_dict={x: batchX, y: batchY})
            loss = cost.eval(feed_dict={x: batchX, y: batchY})
            writer.add_scalar("Training Accuracy", acc, counter)
            writer.add_scalar("Training Loss", loss, counter)
            # print("Iter " + str(step * batchSize) + ", Minibatch Loss= " +
            #       "{:.6f}".format(loss) + ", Training Accuracy= " +
            #       "{:.5f}".format(acc))
            test_acc = accuracy.eval(feed_dict={x: testData, y: testLabel})
            writer.add_scalar("Testing Accuracy", test_acc, counter)
        step += 1

    print('Optimization finished')
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: testData, y: testLabel}))
"""

