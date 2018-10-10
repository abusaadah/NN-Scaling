import MLP1
import profile_formulation1 as pf 
import numpy as np
import tensorflow as tf
import train_test as tt
from tensorflow.python.framework import ops
np.random.seed(1)

## Profile Coefficient
profile_1 = pf.profile_coefficient(0.2, 100)

## Profile Parameters, Loss function & Optimizer
parameters = MLP1.initialize_param(784, 100, 100, 10)
logist_1 = MLP1.mlp(X, profile_1, parameters)
loss_op_1, optimizer_1 = MLP1.optimize_param(logist_1, Y, learning_rate)

## First Profile Training & Saving
saver.save(sess, "my_model"), saver.export_meta_graph("my_model.meta") = tt.train_model(loss_op_1, optimizer_1, logist_1)

## Percentage of the Channels Dropped at the Inference
profile_infr, percentage_channel = pf.prof_inf(0.1, 1.0, 10)

# Drop Weights with profile coefficients at Inference
logist_infr = tt.multilayer_perceptron_drop(X,  profile_infr)

## Test Model 
test_accuracy_infr = tt.model_test(logist_infr)

test_accuracy(test_accuracy_infr)



