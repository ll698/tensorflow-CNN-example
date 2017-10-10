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

# def return_matrix_form(data_list):
#     x = []
#     y = []
#     for d in data_list:
#         data = d[b'data'] / 255
#         labels = d[b'labels']
#         targets = np.array(labels).reshape(-1)
#         labels = np.eye(10)[targets]
#         if x == [] and y == []:
#             x = data
#             y = labels
#
#         else:
#             x = np.concatenate((x,data),axis = 0)
#             y = np.concatenate((y,labels), axis = 0)
#
#     return x, y
#
#
#
# # Load your dataset as a numpy matrix, each row represents an image
#
# #Unpickling our training data
# data_dict0 = unpickle("cifar10/data_batch_1")
# data_dict1 = unpickle("cifar10/data_batch_2")
# data_dict2 = unpickle("cifar10/data_batch_3")
# data_dict3 = unpickle("cifar10/data_batch_4")
# data_dict4 = unpickle("cifar10/data_batch_5")
#
# #unpickeling out testing data
# data_dict_test = unpickle("cifar10/test_batch")
#
# training = [data_dict0,data_dict1,data_dict2,data_dict3,data_dict4]
# testing = [data_dict_test]
#
# trainX, trainY = return_matrix_form(training)
# testX, testY = return_matrix_form(testing)





def cifar_10_reshape(batch_arg):
    output=np.reshape((batch_arg /255),(10000,3,32,32)).transpose(0,2,3,1)
    return output

def unpickle(file):
    import cPickle
    fo=open(file,'rb')
    dict=cPickle.load(fo)
    fo.close()
    return dict




#Loading cifar-10 data and reshaping it to be batch_sizex32x32x3
batch1=unpickle('cifar10/data_batch_1')
batch2=unpickle('cifar10/data_batch_2')
batch3=unpickle('cifar10/data_batch_3')
batch4=unpickle('cifar10/data_batch_4')
batch5=unpickle('cifar10/data_batch_5')



batch1_data=cifar_10_reshape(batch1['data'])
batch2_data=cifar_10_reshape(batch2['data'])
batch3_data=cifar_10_reshape(batch3['data'])
batch4_data=cifar_10_reshape(batch4['data'])
batch5_data=cifar_10_reshape(batch5['data'])

batch1_labels=batch1['labels']
batch2_labels=batch2['labels']
batch3_labels=batch3['labels']
batch4_labels=batch4['labels']
batch5_labels=batch5['labels']

test_batch=unpickle('cifar10/test_batch')
testX = cifar_10_reshape(test_batch['data'])
test_labels_data=test_batch['labels']


trainX = np.concatenate((batch1_data,batch2_data,batch3_data,batch4_data,batch5_data),axis=0)
train_labels_data=np.concatenate((batch1_labels,batch2_labels,batch3_labels,batch4_labels,batch5_labels),axis=0)

#one-hot encodinf of labels
trainY=np.zeros((50000,10),dtype=np.float32)
testY=np.zeros((10000,10),dtype=np.float32)

for i in range(50000):
    a=train_labels_data[i]
    trainY[i,a]=1.0

for j in range(10000):
    b=test_labels_data[j]
    testY[j,b]=1.0





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
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

def normalize(x, norm_noise):
    noise = tf.random_normal(shape=tf.shape(x), mean = 0.0, stddev = norm_noise, dtype=tf.float32)
    return tf.nn.lrn(x + noise, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')


"""
We define our placeholders here. A placeholder is a variable
in which we will assign data to at a later point.
It is useful for defining our graph without requiring the data itself
"""
#placeholders for the input and output layers of our model
x=tf.placeholder(tf.float32, [None, None, None, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)
keep_prob_conv = tf.placeholder(tf.float32)
norm_noise = tf.placeholder(tf.float32)

"""We need to reshape our flat vector representation of an image for the first convolutional layer,
    this converts out 3072 dimensional vector representing an image to a 32 x 32 matrix with a depth of 3,
    this depth is dependent on image color-type. Since we are using rgb color images, we need three channels per color.
    A black and white image would only require one.
    The negative one indicates there can be a variable number for this dimension for our model.
"""
#x_image = tf.reshape(x, [-1,32,32,3])

"""First convolutional layer"""
#first convolutional layer

W_conv1 = weight_tensor([5, 5, 3, 32])

b_conv1 = bias_tensor([32])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

norm1 = normalize(h_conv1, norm_noise)

drop1 = tf.nn.dropout(norm1, keep_prob_conv)



#Second convolutional layer

W_conv2 = weight_tensor([5, 5, 32, 32])

b_conv2 = bias_tensor([32])

h_conv2 = tf.nn.relu(conv2d(drop1, W_conv2) + b_conv2)

norm2 = normalize(h_conv2, norm_noise)

h_pool2 = max_pool_2x2(norm2)

drop2 = tf.nn.dropout(h_pool2, keep_prob_conv)


#Third convolutional layer

W_conv3 = weight_tensor([5, 5, 32, 32])

b_conv3 = bias_tensor([32])


h_conv3 = tf.nn.relu(conv2d(drop2, W_conv3) + b_conv3)

norm3 = normalize(h_conv3, norm_noise)

h_pool3 = max_pool_2x2(norm3)

#Densely connected layer

flattened = tf.reshape(h_pool3, [-1, 32 * 32 * 32])

dim = flattened.get_shape()[1].value

flattened_drop = tf.nn.dropout(flattened, keep_prob)

W_fc1 = weight_tensor([dim, 512])

b_fc1 = bias_tensor([512])





h_fc1 = tf.nn.relu(tf.matmul(flattened_drop, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)





#readout layer

W_fc3 = weight_tensor([512, 10])

b_fc3 = bias_tensor([10])



# y = Wx

y_conv = tf.matmul(h_fc1_drop, W_fc3) + b_fc3


"""Here we are defining our loss function, for this model we are using,
in this example we want to reduce the mean of a cross entropy function"""
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))

"""This defines our optimization method, in this case we are implementing gradient descent
 on a small batch from the dataset, this is also known as stochastic gradient descent. This line can literally be interpreted
 as using gradient descent with our defined learning rate to minimize the cross entropy function"""
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

"""We define our definition of a correct prediction, our model outputs a 10 dimensional vector where each dimension represents the probability of the inputted image being that class,
thus we want to find the max probabiilty in that vector"""
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.device("/gpu:0"):
	sess.run(tf.global_variables_initializer())


"""This is the training component of the neural network"""
print('Training neural network...')
print('training iterations: 100000...')
for i in range(1000000):

    #We pull a batch of samples from our training data to train on
    if i < 10000:
        subset = np.random.randint(50000, size=128)

    if i < 15000:
        subset = np.random.randint(50000, size=256)

    else:
        subset = np.random.randint(50000, size=512)

    batchX = trainX[subset, :]
    batchY = trainY[subset, :]
    index = 0

    #Every 100 training iterations, we evaulated our model on the training data to get a sense of its performance
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batchX, y: batchY, keep_prob: 1.0, keep_prob_conv: 1.0,  norm_noise: 0.0})
        test_subset = np.random.randint(10000, size=3000)
        test_accuracy = accuracy.eval(feed_dict={x:testX[test_subset, :], y: testY[test_subset, :], keep_prob: 1.0, keep_prob_conv: 1.0,  norm_noise: 0.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        print("step %d, testing accuracy %g"%(i, test_accuracy))
        print("running minibatch...")

    if i < 8000:
        train_step.run(feed_dict={x: batchX, y: batchY, keep_prob: 0.5, keep_prob_conv: 0.8, norm_noise: 0.00})

    elif i < 13000:
        train_step.run(feed_dict={x: batchX, y: batchY, keep_prob: 0.5, keep_prob_conv: 0.7, norm_noise: 0.05})

    else:
        train_step.run(feed_dict={x: batchX, y: batchY, keep_prob: 0.5, keep_prob_conv: 0.5, norm_noise: 0.2})



test_accuracy = accuracy.eval(feed_dict={x:testX, y: testY, keep_prob: 1.0, keep_prob_conv: 1.0})
print("testing accuracy: %g" % test_accuracy)
