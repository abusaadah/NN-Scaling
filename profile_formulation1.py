import tensorflow as tf
import numpy as np
import profile_functions1 as pc
import os
 
## Linear Profile function
L = pc.linear_func(100)

def linear_profile(lp, n_1):
    p_L = tf.constant(L, shape = [1, 100])
    L_11 = tf.constant(1.0, shape = [1, int(np.round((lp) * n_1))])
    L_12 = tf.zeros(shape = [1, int(np.round((1 - lp) * n_1))])
    L1 = tf.concat((L_11, L_12), axis = 1)
    p_L1 = tf.multiply(L1, p_L)
    return p_L1

## Profile Coefficient
def profile_coefficient(percentage_idp, n_1):
    p_l = linear_profile(percentage_idp, n_1)
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

