import tensorflow as tf 
import numpy as np 
import pickle
import random
import matplotlib.pyplot as plt
import os, pdb

# Put file path as a string here
CIFAR_DIR = 'cifar-10-batches-py/'

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict

dirs = ['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']
all_data = [0,1,2,3,4,5,6]

for i,direc in zip(all_data,dirs):
    all_data[i] = unpickle(CIFAR_DIR+direc)

batch_meta = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]

## batch_meta
## data_batch1.keys()

##Display a single image using matplotlib

# Put the code here that transforms the X array!
X = data_batch1[b'data']
print(X.shape)
X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
plt.imshow(X[10])

##### Helper Functions for Dealing With Data

def one_hot_encode(vec, vals=10):
    '''
    For use to one-hot encode the 10- possible labels
    '''
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

class CifarHelper():
    
    def __init__(self):
        self.i = 0
        
        # Grabs a list of all the data batches for training
        self.all_train_batches = [data_batch1,data_batch2,data_batch3,data_batch4,data_batch5]
        # Grabs a list of all the test batches (really just one batch)
        self.test_batch = [test_batch]
        
        # Intialize some empty variables for later on
        self.training_images = None
        self.training_labels = None
        
        self.test_images = None
        self.test_labels = None
    
    def set_up_images(self):
        
        print("Setting Up Training Images and Labels")
        
        # Vertically stacks the training images
        self.training_images = np.vstack([d[b"data"] for d in self.all_train_batches])
        train_len = len(self.training_images)
        
        # Reshapes and normalizes training images
        self.training_images = self.training_images.reshape(train_len,3,32,32).transpose(0,2,3,1)/255
        # One hot Encodes the training labels (e.g. [0,0,0,1,0,0,0,0,0,0])
        self.training_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.all_train_batches]), 10)
        
        print("Setting Up Test Images and Labels")
        
        # Vertically stacks the test images
        self.test_images = np.vstack([d[b"data"] for d in self.test_batch])
        test_len = len(self.test_images)
        
        # Reshapes and normalizes test images
        self.test_images = self.test_images.reshape(test_len,3,32,32).transpose(0,2,3,1)/255
        # One hot Encodes the test labels (e.g. [0,0,0,1,0,0,0,0,0,0])
        self.test_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.test_batch]), 10)

        
    def next_batch(self, batch_size):
        # The 100 dimension in the reshape call is set by an assumed batch size of 100
        x = self.training_images[self.i:self.i+batch_size].reshape(100,32,32,3)
        y = self.training_labels[self.i:self.i+batch_size]
        self.i = (self.i + batch_size) #% len(self.training_images)
        return x, y

# Call the CifarHelfer class
ch = CifarHelper()
ch.set_up_images()

# grab the next batch use this line
batch = ch.next_batch(100)


##### Creating the Model

import tensorflow as tf
import numpy as np

'''
x shape = [None,32,32,3]
y_true shape = [None,10]
'''
x=tf.placeholder(tf.float32,shape=[None, 32,32,3])
y_true=tf.placeholder(tf.float32,shape=[None,10])

### Define Profoile Functions and Coefficients

def linear_func(n):
    return [np.float32(1.0 - 1.0 * i/n) for i in range(1, n + 1)]

## Profile Coefficient
def profile_coefficient(lp, n_channel):

    ## Linear Profile function
    L = linear_func(n_channel)  
    p_L = tf.constant(L, shape = [1, n_channel])
    L_11 = tf.constant(1.0, shape = [1, int(np.round((lp) * n_channel))])
    L_12 = tf.zeros(shape = [1, int(np.round((1 - lp) * n_channel))])
    L1 = tf.concat((L_11, L_12), axis = 1)
    p_l = tf.multiply(L1, p_L)
    profile_l = tf.stack(p_l, axis = 0) 
    profile_l = tf.convert_to_tensor(profile_l, dtype = tf.float32)
    return profile_l

## Compute the Percentage of Channels Dropped at the Inference Phase
def prof_inf(min, max, step):
    profile_infr = []
    percentage_channel = np.linspace(min, max, step)
    for i in percentage_channel:
        p_L1 = linear_profile(i, n_channel)
        profile = tf.stack(p_L1, axis = 0) 
        profile_infr.append(profile)
        profile_i = tf.convert_to_tensor(profile_infr, dtype = tf.float32)
    return profile_infr, percentage_channel


'''
Helper Functions
init_weights
init_bias
conv2d
max_pool_2by2
convolutional_layer
normal_full_layer
'''

def init_weights(shape):
    init_random_val=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init_random_val)
def init_bias(shape):
    init_bisa_val=tf.constant(0.1,shape=shape)
    return tf.Variable(init_bisa_val)

def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def conv_layer(x_input,shape):
    W=init_weights(shape)
    b=init_bias([shape[3]])
    return tf.nn.relu(conv2d(x_input,W)+b)

def fully_con_layer_n(layer_input,size):
    input_size=int(layer_input.get_shape()[1])
    W=init_weights([input_size,size])
    b=init_bias([size])
    return tf.nn.relu(tf.matmul(layer_input,W)+b)

def fully_con_layer_final(layer_input,size):
    input_size=int(layer_input.get_shape()[1])
    W=init_weights([input_size,size])
    b=init_bias([size])
    return tf.matmul(layer_input,W)+b

'''
Create the Layers
'''
### First Conv_Layer
conv_11=tf.multiply(conv_layer(x,[3,3,3,64]),profile_coefficient(0.3,64))
conv_11_pooling=max_pool(conv_11)
conv_12=tf.multiply(conv_layer(conv_11_pooling,[3,3,64,64]),profile_coefficient(0.3,64))
conv_12_pooling=max_pool(conv_12)

### Second Conv_Layer
conv_21=tf.multiply(conv_layer(conv_12_pooling,[3,3,64,128]),profile_coefficient(0.3,128))
conv_21_pooling=max_pool(conv_21)
conv_22=tf.multiply(conv_layer(conv_21_pooling,[3,3,128,128]),profile_coefficient(0.3,128))
conv_22_pooling=max_pool(conv_22)

### Third Conv_Layer
conv_31=tf.multiply(conv_layer(conv_22_pooling,[3,3,128,256]),profile_coefficient(0.3,256))
conv_31_pooling=max_pool(conv_31)
conv_32=tf.multiply(conv_layer(conv_31_pooling,[3,3,256,256]),profile_coefficient(0.3,256))
conv_32_pooling=max_pool(conv_32)
conv_33=tf.multiply(conv_layer(conv_32_pooling,[3,3,256,256]),profile_coefficient(0.3,256))
conv_33_pooling=max_pool(conv_33)

### Fourth Conv_Layer
conv_41=tf.multiply(conv_layer(conv_33_pooling,[3,3,256,512]),profile_coefficient(0.3, 512))
conv_41_pooling= max_pool(conv_41)
conv_42=tf.multiply(conv_layer(conv_41_pooling,[3,3,512,512]),profile_coefficient(0.3, 512))
conv_42_pooling= max_pool(conv_42)
conv_43=tf.multiply(conv_layer(conv_42_pooling,[3,3,512,512]),profile_coefficient(0.3, 512))
conv_43_pooling= max_pool(conv_43)

### Fifth Conv_Layer
conv_51=tf.multiply(conv_layer(conv_43_pooling,[3,3,512,512]),profile_coefficient(0.3, 512))
conv_51_pooling= max_pool(conv_51)
conv_52=tf.multiply(conv_layer(conv_51_pooling,[3,3,512,512]),profile_coefficient(0.3, 512))
conv_52_pooling= max_pool(conv_52)
conv_53=tf.multiply(conv_layer(conv_52_pooling,[3,3,512,512]),profile_coefficient(0.3, 512))
conv_53_pooling= max_pool(conv_53)

#### Now create a flattened layer by reshaping the pooling layer
#conv_53_flat=tf.reshape(conv_53_pooling,[-1,7*7*512])
conv_53_flat=tf.reshape(conv_53_pooling,[-1,51200])


print(conv_53_flat)

### Fully Conected layers
fully_con_1=tf.multiply(fully_con_layer_n(conv_53_flat,4096),profile_coefficient(0.3,4096))
fully_con_2=tf.multiply(fully_con_layer_n(fully_con_1,4096),profile_coefficient(0.3,4096))
y_pred=fully_con_layer_final(fully_con_2,10)
print(y_pred.shape)

### Create a cross_entropy loss function
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y_pred))

optimizer=tf.train.AdamOptimizer(learning_rate=0.001)
train=optimizer.minimize(cross_entropy)

### Create a variable to intialize all the global tf variables.
init=tf.global_variables_initializer()

### Graph Session
with tf.Session() as sess:
    sess.run(init)
    
    for i in range(100):
        batch=ch.next_batch(100)
        
        sess.run(train,feed_dict={x:batch[0],y_true:batch[1]})
        
        if i % 100 == 0:
            print('step: {}'.format(i))
            correct_pred=tf.equal(tf.arg_max(y_true,1),tf.arg_max(y_pred,1))
            acc=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
            print(sess.run(acc,feed_dict={x:ch.test_images,y_true:ch.test_labels}))
            print('\n')
            
            
            
            
            
            
            
    