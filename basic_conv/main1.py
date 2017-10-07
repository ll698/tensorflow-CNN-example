from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random

sess = tf.InteractiveSession()




"""This part deals with setting up your data as a matrix usable for training"""

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def return_matrix_form(data_list):
    x = []
    y = []
    for d in data_list:
        data = d[b'data'] / 255
        labels = d[b'labels']
        targets = np.array(labels).reshape(-1)
        labels = np.eye(10)[targets]
        if x == [] and y == []:
            x = data
            y = labels

        else:
            x = np.concatenate((x,data),axis = 0)
            y = np.concatenate((y,labels), axis = 0)

    return x, y



# Load your dataset as a numpy matrix, each row represents an image

#Unpickling our training data
data_dict0 = unpickle("cifar10/data_batch_1")
data_dict1 = unpickle("cifar10/data_batch_2")
data_dict2 = unpickle("cifar10/data_batch_3")
data_dict3 = unpickle("cifar10/data_batch_4")

#unpickeling out testing data
data_dict_test = unpickle("cifar10/data_batch_5")

training = [data_dict0]
testing = [data_dict_test]

trainX, trainY = return_matrix_form(training)
testX, testY = return_matrix_form(testing)


"""Define our weight initialization function, our weights are drawn
from a truncated normal distribution with a mean of 0 and a standard deviation of 0.1"""
def weight_tensor(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_tensor(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    norm = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    return tf.nn.max_pool(norm, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

"""
We define our placeholders here. A placeholder is a variable
in which we will assign data to at a later point.
It is useful for defining our graph without requiring the data itself
"""
#placeholders for the input and output layers of our model
x = tf.placeholder(tf.float32, shape=[None, 3072])
y = tf.placeholder(tf.float32, shape=[None, 10])

"""We need to reshape our flat vector representation of an image for the first convolutional layer,
    this converts out 3072 dimensional vector representing an image to a 32 x 32 matrix with a depth of 3,
    this depth is dependent on image color-type. Since we are using rgb color images, we need three channels per color.
    A black and white image would only require one.
    The negative one indicates there can be a variable number for this dimension for our model.
"""
x_image = tf.reshape(x, [-1,32,32,3])

"""First convolutional layer"""

#placholders for the first convolutional layer
W_conv1 = weight_tensor([5, 5, 3, 64])
b_conv1 = bias_tensor([64])

#input convolutional layer
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Second convolutional layer
W_conv2 = weight_tensor([5, 5, 64, 64])
b_conv2 = bias_tensor([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Densely connected layer
W_fc1 = weight_tensor([8*8*64, 128])
b_fc1 = bias_tensor([128])

h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Second densely connected layer
W_fc2 = weight_tensor([128, 128])
b_fc2 = bias_tensor([128])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

#readout layer
W_fc3 = weight_tensor([128, 10])
b_fc3 = bias_tensor([10])

# y = Wx
y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3






"""Here we are defining our loss function, for this model we are using,
in this example we want to reduce the mean of a cross entropy function"""
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))

"""This is our learning rate, it helps use decide our step size for gradient updates,
This is helpful because we can initialize with large steps in the function space and minimize it,
allowing us quickly find an optimal solution that minizes out loss"""
learning_rate = tf.train.exponential_decay(0.1, 25,
                                           10000, 0.0001, staircase=True)

"""This defines our optimization method, in this case we are implementing gradient descent
 on a small batch from the dataset, this is also known as stochastic gradient descent. This line can literally be interpreted
 as using gradient descent with our defined learning rate to minimize the cross entropy function"""
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

"""We define our definition of a correct prediction, our model outputs a 10 dimensional vector where each dimension represents the probability of the inputted image being that class,
thus we want to find the max probabiilty in that vector"""
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())


"""This is the training component of the neural network"""
print('Training neural network...')
print('training iterations: 1000...')
for i in range(10000):

    #We pull a batch of samples from our training data to train on
    subset = np.random.randint(10000, size=32)
    batchX = trainX[subset, :]
    batchY = trainY[subset, :]
    index = 0

    #Every 100 training iterations, we evaulated our model on the training data to get a sense of its performance
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batchX, y: batchY})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        print("running minibatch...")

    train_step.run(feed_dict={x: batchX, y: batchY})

test_accuracy = accuracy.eval(feed_dict={x:testX, y: testY})
print("testing accuracy: %g" % test_accuracy)
