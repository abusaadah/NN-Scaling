
import tensorflow as tf
import numpy as np 
import os
import matplotlib.pyplot as plt 
import preprocess_data
from preprocess_data import valid_features, valid_labels, test_features, test_labels
import pickle

epochs = 5
batch_size = 128
learning_rate = 0.0001

def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

def load_preprocess_training_batch(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)

# Inputs
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
y =  tf.placeholder(tf.float32, shape=(None, 10), name='output_y')

## Linear Function
def linear_func(n):
    return[np.float32(1.0 - 1.0 * i/n) for i in range(1, n + 1)]

## Linear Profile function


def linear_profile(lp, n_channel):
    L = linear_func(n_channel)
    p_L = tf.constant(L, shape = [1, n_channel])
    L_11 = tf.constant(1.0, shape = [1, int(np.round((lp) * n_channel))])
    L_12 = tf.zeros(shape = [1, int(np.round((1 - lp) * n_channel))])
    L1 = tf.concat((L_11, L_12), axis = 1)
    p_L1 = tf.multiply(L1, p_L)
    return p_L1

## Profile Coefficient
def profile_coefficient(percentage_idp, n_channel):
    p_l = linear_profile(percentage_idp, n_channel)
    profile_l = tf.stack(p_l, axis = 0) 
    profile_l = tf.convert_to_tensor(profile_l, dtype=tf.float32)
    return  profile_l

## Compute the Percentage of Channels Dropped at the Inference Phase
def prof_inf(min, max, step):
    profile_infr = []
    percentage_channel = np.linspace(min, max, step)
    for i in percentage_channel:
        p_L1 = linear_profile(i, 100)
        profile = tf.stack(p_L1, axis = 0) 
        profile_infr.append(profile)
        profile_i = tf.convert_to_tensor(profile_infr, dtype=tf.float32)
    return profile_infr, percentage_channel


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


def conv_net(x, percentage_idp):
    ### First Conv_Layer
    conv_11=tf.multiply(conv_layer(x,[3,3,3,64]),profile_coefficient(percentage_idp,64))
    conv_11_pooling=max_pool(conv_11)
    conv_12=tf.multiply(conv_layer(conv_11_pooling,[3,3,64,64]),profile_coefficient(percentage_idp,64))
    conv_12_pooling=max_pool(conv_12)

    ### Second Conv_Layer
    conv_21=tf.multiply(conv_layer(conv_12_pooling,[3,3,64,128]),profile_coefficient(percentage_idp,128))
    conv_21_pooling=max_pool(conv_21)
    conv_22=tf.multiply(conv_layer(conv_21_pooling,[3,3,128,128]),profile_coefficient(percentage_idp,128))
    conv_22_pooling=max_pool(conv_22)

    ### Third Conv_Layer
    conv_31=tf.multiply(conv_layer(conv_22_pooling,[3,3,128,256]),profile_coefficient(percentage_idp,256))
    conv_31_pooling=max_pool(conv_31)
    conv_32=tf.multiply(conv_layer(conv_31_pooling,[3,3,256,256]),profile_coefficient(percentage_idp,256))
    conv_32_pooling=max_pool(conv_32)
    conv_33=tf.multiply(conv_layer(conv_32_pooling,[3,3,256,256]),profile_coefficient(percentage_idp,256))
    conv_33_pooling=max_pool(conv_33)

    ### Fourth Conv_Layer
    conv_41=tf.multiply(conv_layer(conv_33_pooling,[3,3,256,512]),profile_coefficient(percentage_idp, 512))
    conv_41_pooling= max_pool(conv_41)
    conv_42=tf.multiply(conv_layer(conv_41_pooling,[3,3,512,512]),profile_coefficient(percentage_idp, 512))
    conv_42_pooling= max_pool(conv_42)
    conv_43=tf.multiply(conv_layer(conv_42_pooling,[3,3,512,512]),profile_coefficient(percentage_idp, 512))
    conv_43_pooling= max_pool(conv_43)

    ### Fifth Conv_Layer
    conv_51=tf.multiply(conv_layer(conv_43_pooling,[3,3,512,512]),profile_coefficient(percentage_idp, 512))
    conv_51_pooling= max_pool(conv_51)
    conv_52=tf.multiply(conv_layer(conv_51_pooling,[3,3,512,512]),profile_coefficient(percentage_idp, 512))
    conv_52_pooling= max_pool(conv_52)
    conv_53=tf.multiply(conv_layer(conv_52_pooling,[3,3,512,512]),profile_coefficient(percentage_idp, 512))
    conv_53_pooling= max_pool(conv_53)

    # Flatten the Layer
    conv_53_flat = tf.contrib.layers.flatten(conv_53_pooling) 

    ### Fully Conected layers
    fully_con_1=tf.multiply(fully_con_layer_n(conv_53_flat,4096),profile_coefficient(percentage_idp,4096))
    fully_con_2=tf.multiply(fully_con_layer_n(fully_con_1,4096),profile_coefficient(percentage_idp,4096))
    fully_con_3=tf.multiply(fully_con_layer_n(fully_con_2,4096),profile_coefficient(percentage_idp,4096))
    out=fully_con_layer_final(fully_con_3,10)
    return out

logits = conv_net(x, 0.3)
model = tf.identity(logits, name='logits') # Name logits Tensor, so that can be loaded from disk after training

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

save_model_path = './image_classification'

def train_neural_network(session, optimizer, feature_batch, label_batch):
    session.run(optimizer, 
                feed_dict={
                    x: feature_batch,
                    y: label_batch,
                })

def print_stats(session, feature_batch, label_batch, cost, accuracy):
    loss = sess.run(cost, 
                    feed_dict={
                        x: feature_batch,
                        y: label_batch
                    })
    valid_acc = sess.run(accuracy, 
                         feed_dict={
                             x: valid_features,
                             y: valid_labels
                         })
    
    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))

save_model_path = './image_classification'

## Inference
percentage_channel = np.linspace(0.1, 1.0, 10)
logist_infr = []

for j in (percentage_channel):
    logist_inf = conv_net(x, j)
    logist_infr.append(logist_inf)
       

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, batch_features, batch_labels)
                
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)
            
    # # Save Model
    # saver = tf.train.Saver()
    # save_path = saver.save(sess, save_model_path)

    ## Test Model 
    test_accuracy_infr = []
    for k in range(len(logist_infr)):
        correct_pred_infr = tf.equal(tf.argmax(logist_infr[k], 1), tf.argmax(y, 1))
        accuracy_infr = tf.reduce_mean(tf.cast(correct_pred_infr, tf.float32), name='test_accuracy')
        test_accuracy_in = accuracy_infr.eval({x: test_features, y: test_labels})*100
        test_accuracy_infr.append(test_accuracy_in)

    for l in percentage_channel:
    	if l % display_step == 0: 
    		print("Percentage_Profile Test Accuracy = {} ".format(test_accuracy_infr))
   

