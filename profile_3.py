import MLP1
import profile_formulation1 as pf 
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# Network Parameters
n_1 = 100               # 1st layer number of neurons
n_2 = 100               # 2nd layer number of neurons
n_input = 784           #MNIST data input (img shape: 28*28)
n_classes = 10          # MNIST total classes (0-9 digits)
learning_rate = 0.001
training_epochs = 40
batch_size = 40
display_step = 1


## Profile Coefficient
graph = tf.Graph()
with graph.as_default():
    profile_3 = pf.profile_coefficient(0.6, 100)
    X = tf.placeholder(tf.float32, [None, n_input])
    Y = tf.placeholder(tf.float32, [None, n_classes])
    parameters = MLP1.initialize_param(784, 100, 100, 10)
    logist_3 = MLP1.mlp(X, profile_3, parameters)
    loss_op_3, optimizer_3 = MLP1.optimize_param(logist_3, Y, learning_rate)
    new2_saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:

    new2_saver.restore(sess,tf.train.latest_checkpoint('./'))

    # Training Loop
    cost_3 = []
    for epoch in range(training_epochs):
        avg_cost3 = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
                
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            
            # Run optimization op (backprop) and cost op (to get loss value)
            c_3,_ = sess.run([loss_op_3, optimizer_3], feed_dict = {X: batch_x, Y: batch_y})
                    
            # Compute average losses
            avg_cost3 += c_3 / total_batch
            cost_3.append(avg_cost3)
            if i % 5000 == 0:
                    pred = tf.nn.softmax(logist_3)  # Apply softmax to logits
                    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    trian_accuracy = accuracy.eval({X: mnist.train.images, Y: mnist.train.labels})   
                       
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch_3:", '%03d' % (epoch + 1), "cost = {:.9f}".format(avg_cost3))
        
    # new2_saver.save(sess, "my_model3") 
    # new2_saver.export_meta_graph("my_model3.meta")
 
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
                Test_Accuracy = ','.join(['{:.0f}'.format(ii) for ii in (test_accuracy_infr)])
                return print("Percentage_Profile3 Test Accuracy = " +Test_Accuracy)
    test_accuracy(test_accuracy_infr)

sess.close()

print("Third Profile Training with both Fisrt & Second Profiles fixed finished")
        

    
        