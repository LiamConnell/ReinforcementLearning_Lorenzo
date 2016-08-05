import tensorflow as tf
import numpy as np
import numpy.random as rng
import pandas.io.data as web
import numpy as np
import pandas as pd

class Model():
    def __init__(self, config, training=True):
        #CONFIG
        input_size = self.input_size =  config.input_size
        output_size = self.output_size =  config.output_size
        num_steps = self.num_steps =  config.num_steps
        
        num_samples = self.num_samples =  config.num_samples
        learning_rate = self.learning_rate =  config.learning_rate
        hidden_size = self.hidden_size =  config.hidden_size
        num_layers = self.num_layers =  config.num_layers
        forget_bias = self.forget_bias = config.forget_bias


        # define placeholders 
        self.environment_ = tf.placeholder(tf.float32, [None, self.input_size])

        #define cell
        cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=self.forget_bias)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * config.num_layers)
        cell = tf.nn.rnn_cell.InputProjectionWrapper(cell,  self.input_size)
        cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, self.output_size)
        self.cell = cell

        #iterate through timesteps
        self.LSTM_iter()     # assign self.final_state and self.outputs
                
        self.sample_policy() # assign self.outputs_softmax, self.sample, self.training_target_cols
        
        self.get_status()    # assign self.status

        # compute the REINFORCEMENT of the sampled decisions
        ones = tf.ones_like(self.training_target_cols)
        reinforcement_ = tf.nn.sigmoid_cross_entropy_with_logits(self.training_target_cols, ones)
        self.reinforcement = tf.transpose(tf.reshape(reinforcement_, [-1,2,num_samples]), [0,2,1]) #[?,5,2]
        
        # multiply reinforcement by reward in order to get a cost function        
        self.reward = tf.mul(self.reinforcement , tf.expand_dims(self.status, -1))
        # minimize cost function
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.reward)

        
    def LSTM_iter(self):
        self.outputs=[]
        self.initial_state = self.cell.zero_state(1, tf.float32)    #batch size?
        self.state = self.initial_state
            
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):  
                if time_step > 0: tf.get_variable_scope().reuse_variables()    
                inp = self.environment_[time_step,:]
                inp = tf.reshape(inp, [1,-1])
                (cell_output, self.state) = self.cell(inp, self.state)
                self.outputs.append(cell_output) 

        self.final_state = self.state 
        self.outputs = tf.reshape(tf.concat(1, self.outputs), [-1, self.output_size])
        
    def sample_policy(self):
        self.outputs_softmax = tf.nn.softmax(self.outputs)
        self.sample = tf.multinomial(tf.log(self.outputs), self.num_samples)
        
        relevant_target_column = {}
        for sample_iter in range(self.num_samples):
            sample_n = self.sample[:,sample_iter]
            sample_mask = tf.cast(tf.reshape(tf.one_hot(sample_n, self.output_size), 
                                             [-1,self.output_size]), tf.float32)
            relevant_target_column[sample_iter] = tf.reduce_sum(
                                                self.outputs_softmax * sample_mask,1)
        self.training_target_cols = tf.concat(1, [tf.reshape(t, [-1,1]) for t in relevant_target_column.values()])
        
    def get_status(self)
        self.status = tf.reduce_sum(self.training_target_cols)
        return