
import tensorflow as tf
import numpy as np
from random import *
import matplotlib.pyplot as plt

# x : 입력값, w : 필터
def convLayer(x, w):
  return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def poolingLayer(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def RELU(x):
  return tf.nn.relu(x)

def batchNormalization(x, is_training):
  return tf.layers.batch_normalization(x, scope='bn', training=is_training)

def affine(x, w, b):
  # using matrix "broadcast" -> b would be broadcasted
  return tf.matmul(x, w) + b

# softmax with cross entropy
def loss(x, y):
  return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=x))

def CNN(x, keep_prob, is_training):
  # convolution filters
  f1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
  f2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
  f3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
  f4 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
  
  # weights for affine layers
  w1 = tf.Variable(tf.random_normal([4 * 4 * 128, 1000]))
  w2 = tf.Variable(tf.random_normal([1000, 100]))
  w3 = tf.Variable(tf.random_normal([100, 10]))
  
  # biases for affine layers
  b1 = tf.Variable(tf.zeros([1, 1000]))
  b2 = tf.Variable(tf.zeros([1, 100]))
  b3 = tf.Variable(tf.zeros([1, 10]))
  
  #1. conv - pool 
  conv1 = convLayer(x, f1)
  relu1 = RELU(conv1)
  pool1 = poolingLayer(relu1) # (?, 14, 14, 32)
  pool1 = tf.nn.dropout(pool1, keep_prob)
  
  #2. conv - pool 
  conv2 = convLayer(pool1, f2)
  relu2 = RELU(conv2)
  pool2 = poolingLayer(relu2) # (?, 7, 7, 64)
  pool2 = tf.nn.dropout(pool2, keep_prob)
  
  #3. conv - pool
  conv3 = convLayer(pool2, f3)
  relu3 = RELU(conv3)
  pool3 = poolingLayer(relu3) # (?, 4, 4, 128)
  pool3 = tf.nn.dropout(pool3, keep_prob)

  #3. affine (input)
  pool3 = tf.reshape(pool3, [-1, 4 * 4 * 128])
  affn1 = affine(pool3, w1, b1) # (?, 1000)
  #btnrm1 = batchNormalization(affn1, is_training)
  relu4 = RELU(affn1)
  relu4 = tf.nn.dropout(relu4, keep_prob)
  
  #4. affine (hidden)
  affn2 = affine(relu4, w2, b2) # (?, 100)
  #btnrm2 = batchNormalization(affn2, is_training)
  relu5 = RELU(affn2)
  relu5 = tf.nn.dropout(relu5, keep_prob)
  
  #5. affine (output)
  affn3 = affine(relu5, w3, b3) # (?, 10)
  
  return affn3

(x_train, t_train), (x_test, t_test) = tf.keras.datasets.fashion_mnist.load_data()

batch_size = 100
learning_rate = 0.001
train_num = 100000

x_train = x_train.reshape(-1, 28, 28, 1)
t_train = tf.one_hot(t_train, 10)
x_test = x_test.reshape(-1, 28, 28, 1)
t_test = tf.one_hot(t_test, 10)

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [batch_size, 10])
keep_prob = tf.placeholder("float")
is_training = tf.placeholder(tf.bool, name='phase')

cnn = CNN(X, keep_prob, is_training)

# make the result to one-hot coding
predict = tf.argmax(cnn, 1)

loss = loss(cnn, Y)

#update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  
#with tf.control_dependencies(update_ops):
train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  
  # make tf object to numpy array
  t_train = t_train.eval()
  t_test = t_test.eval()
  
  # make it to one-hot coding
  t_test = np.argmax(t_test, axis=1)

  x_loss = range(train_num)
  y_loss_values = np.array([])

  y_acc_values = np.array([])
  epoch = 0
  
  for i in range(train_num):
    rand = np.random.choice(60000, batch_size)

    _, loss_value = sess.run([train_optimizer, loss], feed_dict={X: x_train[rand], Y: t_train[rand], keep_prob: 0.8, is_training: True})
    y_loss_values = np.append(y_loss_values, [loss_value], axis=0)

    if i % 1000 == 0:
      epoch += 1
      
      predict_label = sess.run(predict, feed_dict={X: x_test, keep_prob: 1.0, is_training: False})
      
      accuracy = np.sum(predict_label == t_test) / float(predict_label.shape[0]) * 100
      y_acc_values = np.append(y_acc_values, [accuracy], axis=0)
      print(accuracy)
      x_acc = range(0, epoch)
      y_acc = y_acc_values[x_acc]
      plt.plot(x_acc, y_acc)
      plt.show()
      

  y_loss = y_loss_values[x_loss]
  plt.plot(x_loss, y_loss)
  plt.show()
  
    
  #print(sess.run(cnn, feed_dict={X: x_train[:batch_size]}).shape)


  
