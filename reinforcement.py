import tensorflow as tf
import numpy as np
import numpy.random as rng
import pandas.io.data as web
import numpy as np
import pandas as pd

from reinforcement_lstm_model import Model

class SmallConfig(object):
    """Small config."""
    input_size = 5
    output_size = 5
    num_steps = 20

    num_samples = 2
    learning_rate = 0.5
    hidden_size = 21
    num_layers = 2
    forget_bias = 0.1
    

# LOAD model and INITIALIZE variables
sess = tf.InteractiveSession()
# initialize variables to random values
with tf.variable_scope("model", reuse=None):
    m = Model(config = SmallConfig)
with tf.variable_scope("model", reuse=True):
    mvalid = Model(config = SmallConfig, training=False)
sess.run(tf.initialize_all_variables())

# import data (toy example)
inputs = np.random.random([20,5])


for epoch in range(200):
    sess.run(m.optimizer, feed_dict={m.environment_: inputs})
    if np.sqrt(epoch+1)%1== 0:
        c = sess.run([ mvalid.cost], feed_dict={mvalid.environment_: inputs})
        c = np.mean(c)
        print("Epoch:", '%04d' % (epoch+1), "cost=",c, )

