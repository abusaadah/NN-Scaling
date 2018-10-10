import numpy as np
import tensorflow as tf
import profile_functions1 as pc
from profile_formulation1 import linear_profile as lp
#from tensorflow.examples.tutorials.mnist import input_data
import math
import matplotlib.pyplot as plt
np.random.seed(1)

n_1 = 100               # 1st layer number of neurons
n_2 = 100               # 2nd layer number of neurons
n_input = 784           #MNIST data input (img shape: 28*28)
n_classes = 10          # MNIST total classes (0-9 digits)

# Store layers weight & bias
def initialize_param(n_input, n_1, n_2, n_class):
    tf.set_random_seed(1)
    W1 = tf.get_variable("W1", shape = [n_input, n_1], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", shape = [n_1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", shape = [n_1, n_2], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", shape = [n_2], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", shape = [n_2, n_class], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", shape = [n_class], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2,"W3": W3,"b3": b3}
    return parameters

def mlp(x, profile_type, parameters):
    Z_ML1 = tf.add(tf.matmul(x, parameters['W1']), parameters['b1'])  
    A_ML1 = tf.nn.relu(Z_ML1)
    P_ML1 = tf.multiply(profile_type, A_ML1)
    Z_ML2 = tf.add(tf.matmul(P_ML1, parameters['W2']), parameters['b2'])  
    A_ML2 = tf.nn.relu(Z_ML2)
    P_ML2 = tf.multiply(profile_type, A_ML2)
    out_layer = tf.add(tf.matmul(P_ML2, parameters['W3']), parameters['b3'])
    return out_layer

def optimize_param(logits, y, learning_rate):
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y))
    #optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = 0.98).minimize(loss_op)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss_op)  
    return loss_op, optimizer



