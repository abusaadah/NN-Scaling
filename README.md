# NN-Scaling
Training a Deep Neural Network (DNN) requires a lot of data and computing resources, because of the size of the network and the number  of training parameters involved. In this project, we design a multiple channel layering framework that uses percentage of the channels (0-20%, 20%-40% and 40%-100%) by multiplying the weights with a profile function that dynamically scale down the model and speed up the training. Finally, the dynamic range across entire channels that gives the best performance at the inference is selected. 

# Note:
  Different weight profile functions could be used, but in this work, we use Linear Profile Function.

# Required packages
  . Tensorflow
  . numpy
  . matplotlib
  . scipy

# How the Code works

  # a) There are six files:
    i. MPL.py: this contains the main model structure
    ii. profile_formulation1.py: common functions that compute channels wieghts for all the profiles 
    iii.  profile_function1.py: contains different profile functions
    iv. train_test.py: Firsr Profile 
    v.  pofile_22.py: Second Profile
    vi.  pofile_3.py: Third Profile
    vii.  plot.py: plots model performance
  
  # b) Run the Code: run 
       i. $ python train_test.py
       ii.  $ python pofile_22.py
       iii $ python plot.py
