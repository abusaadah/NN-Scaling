import MLP1
import profile_formulation1 as pf 
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
np.random.seed(1)

# Network Parameters
n_1 = 100               # 1st layer number of neurons
n_2 = 100               # 2nd layer number of neurons
n_input = 784           #MNIST data input (img shape: 28*28)
n_classes = 10          # MNIST total classes (0-9 digits)
learning_rate = 0.001
training_epochs = 40
batch_size = 40
display_step = 1

# tf Graph input
tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_classes])

## Profile Coefficient
profile_1 = pf.profile_coefficient(0.2, 100)

parameters = MLP1.initialize_param(784, 100, 100, 10)
logist_1 = MLP1.mlp(X, profile_1, parameters)
loss_op_1, optimizer_1 = MLP1.optimize_param(logist_1, Y, learning_rate)
    
# Initializing the variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    def train_model(loss_type, optimizer_type, logist_type):
        
        ## Creating a Saver Method 
        saver = tf.train.Saver()
    
        # Training Loop
        cost = []
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
                
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
            
                # Run optimization op (backprop) and cost op (to get loss value)
                c,_ = sess.run([loss_type, optimizer_type], feed_dict = {X: batch_x, Y: batch_y})
                    
                # Compute average losses
                avg_cost += c / total_batch
                cost.append(avg_cost)            
                if i % 5000 == 0:
                    pred = tf.nn.softmax(logist_type)  # Apply softmax to logits
                    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    trian_accuracy = accuracy.eval({X: mnist.train.images, Y: mnist.train.labels})   
        
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%03d' % (epoch + 1), "cost = {:.9f}".format(avg_cost))
     
        ## Saving the First Profile Model & Generate MetaGrapgDef
        return saver.save(sess, "my_model"), saver.export_meta_graph("my_model.meta")
    train_model(loss_op_1, optimizer_1, logist_1)

    ## Inference Phase
    profile_infr, percentage_channel = pf.prof_inf(0.1, 1.0, 20)
    
    # Drop Weights with profile coefficients at Inference
    def multilayer_perceptron_drop(x, profile_in = profile_infr):
        logist_infr = []
        for j in range(len(percentage_channel)):
            Z_1 = tf.add(tf.matmul(x, parameters['W1']), parameters['b1'])  
            A_1 = tf.nn.relu(Z_1)
            P_1 = tf.multiply(profile_in[j], A_1)
            Z_2 = tf.add(tf.matmul(P_1, parameters['W2']), parameters['b2'])  
            A_2 = tf.nn.relu(Z_2)
            P_2 = tf.multiply(profile_in[j], A_2)
            out_layer = tf.add(tf.matmul(P_2, parameters['W3']), parameters['b3'])
            logist_infr.append(out_layer)
        return   logist_infr    
    logist_infr = multilayer_perceptron_drop(X,  profile_infr)

    ## Test Model 
    def model_test(logist_in):
        test_accuracy_infr = []
        for k in range(len(logist_infr)):
            pred_infr = tf.nn.softmax(logist_infr[k])
            correct_prediction_infr = tf.equal(tf.argmax(pred_infr, 1), tf.argmax(Y, 1))
            accuracy_infr = tf.reduce_mean(tf.cast(correct_prediction_infr, "float"))
            test_accuracy_in = accuracy_infr.eval({X: mnist.test.images, Y: mnist.test.labels})*100
            test_accuracy_infr.append(test_accuracy_in)
        return test_accuracy_infr
        sess.close()
    test_accuracy_infr = model_test(logist_infr)

    def test_accuracy(test_accuracy_infr):
        for l in percentage_channel:
            if l % display_step == 0: 
                #Test_Accuracy = ','.join([format(ii, '.2f') for ii in test_accuracy_infr])
                #Test_Accuracy = ','.join(['{:.4f}'.format(ii) for ii in (test_accuracy_infr)])
                #return print("Percentage_Profile Test Accuracy = " +Test_Accuracy)
                return print("Percentage_Profile Test Accuracy = {} ".format(test_accuracy_infr))
    test_accuracy(test_accuracy_infr)
