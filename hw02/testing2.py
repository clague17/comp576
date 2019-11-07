import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

import matplotlib.pyplot as plt

# COMP 576 Assignment 2 Task 3
# Grace Tan
# gzt1

#call mnist function
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learningRate = 1e-3
trainingIters = 100000
batchSize = 128
displayStep = 10

nInput = 28		#we want the input to take the 28 pixels
nSteps = 28		#every 28
nHidden = 256	#number of neurons for the RNN
nClasses = 10	#this is MNIST so you know

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = {
	'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
	'out': tf.Variable(tf.random_normal([nClasses]))
}

def RNN(x, weights, biases, gru=False):
	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, nInput])
	x = tf.split(value=x, num_or_size_splits=nSteps, axis=0) #configuring so you can get it as needed for the 28 pixels

	#find which lstm to use in the documentation
	if gru:
		lstmCell = rnn.GRUCell(nHidden)
	else:
		lstmCell = rnn.BasicLSTMCell(nHidden)

	#for the rnn where to get the output and hidden state
	outputs, states = tf.nn.static_rnn(lstmCell, x, dtype=tf.float32)

	return tf.matmul(outputs[-1], weights['out'])+ biases['out']

pred = RNN(x, weights, biases, gru=True)

#optimization
#create the cost, optimization, evaluation, and accuracy
#for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

correctPred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

init = tf.initialize_all_variables()

train_accuracy = []
test_accuracy = []
train_loss = []

testData = mnist.test.images.reshape((-1, nSteps, nInput))
testLabel = mnist.test.labels

with tf.Session() as sess:
	sess.run(init)
	step = 1

	while step* batchSize < trainingIters:
		#mnist has a way to get the next batch
		batchX, batchY = mnist.train.next_batch(batchSize)
		batchX = batchX.reshape((batchSize, nSteps, nInput))

		sess.run(optimizer, feed_dict={x: batchX, y: batchY})

		if step % displayStep == 0:
			# training
			acc = accuracy.eval(feed_dict={x: batchX, y: batchY})
			loss = cost.eval(feed_dict={x: batchX, y: batchY})
			train_accuracy.append(acc)
			train_loss.append(loss)
			print("Iter " + str(step*batchSize) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

			# testing
			test_acc = accuracy.eval(feed_dict={x: testData, y: testLabel})
			test_accuracy.append(test_acc)
			print("Test Accuracy= " + "{:.5f}".format(test_acc))

		step +=1
	print('Optimization finished')

	testData = mnist.test.images.reshape((-1, nSteps, nInput))
	testLabel = mnist.test.labels
	print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: testData, y: testLabel}))

# plot train and test accuracy
plt.plot(range(len(train_accuracy)), train_accuracy, 'b', label="Training Accuracy")
plt.plot(range(len(test_accuracy)), test_accuracy, "r", label="Testing Accuracy")
plt.title("Training and Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss ")
plt.legend()
plt.figure()
plt.show()

# plot train loss
plt.plot(range(len(train_loss)), train_loss, 'b', label="Training loss")
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.figure()
plt.show()