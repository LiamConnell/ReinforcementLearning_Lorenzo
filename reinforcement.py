import tensorflow as tf
import numpy as np
import numpy.random as rng
import pandas.io.data as web
import numpy as np
import pandas as pd
import argparse
import os
import pickle as cPickle

from reinforcement_lstm_model import Model

def parser():
    parser = argparse.ArgumentParser()
    
    
    
    parser.add_argument('--data_file', type=str, default=None,
                       help='data file path')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                            'config.pkl'        : configuration;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    
                        
    parser.add_argument('--input_size', type=int, default=5,
                       help='size of input vector for each time step')
    parser.add_argument('--output_size', type=int, default=5,
                       help='dimension of vector the LSTM should output each time step')
    parser.add_argument('--num_steps', type=int, default=20,
                       help='number of steps to go into the LSTM at a time')
    
    parser.add_argument('--num_samples', type=int, default=2,
                       help='number of samples to make during trainging')
    parser.add_argument('--learning_rate', type=float, default=0.5,
                       help='learning rate for gradient descent')
    parser.add_argument('--hidden_size', type=int, default=21, 
                       help='size of LSTM hidden state')                                       
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the LSTM')
    parser.add_argument('--forget_bias', type=float, default=0.1,
                       help='forget bias of LSTM')
    
    
    parser.add_argument('--num_epochs', type=int, default=500,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=100,
                       help='save frequency')
    
    
                          
    
    args = parser.parse_args()
    return args
    
    
def read_data_fn(file_path):
    '''modify this fn to correctly format data'''
    data = pd.read_csv(file_path)
    return data
    

def main_example():
    args = parser()
    
    
    # check compatibility if training is continued from previously saved model (taken from [1])
    if args.init_from is not None:
        # check if all necessary files exist 
        assert os.path.isdir(args.init_from)," %s must be an existing path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt,"No checkpoint found"
        assert ckpt.model_checkpoint_path,"No model path found in checkpoint"
        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same=["hidden_size","num_layers"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme
    # save config args
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    
    
    # LOAD model and INITIALIZE variables
    sess = tf.InteractiveSession()
    with tf.variable_scope("model", reuse=None):
        m = Model(config = args)
    with tf.variable_scope("model", reuse=True):
        mvalid = Model(config = args, training=False)
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    
    # RESTORE checkpointed model if asked
    if args.init_from is not None:
        saver.restore(sess, ckpt.model_checkpoint_path)
        
    # import data 
    if args.data_file==None: #(toy example)
        inputs = np.random.random([args.num_steps, args.input_size])
    else:
        inputs = read_data_fn(args.data_file)


    for epoch in range(args.num_epochs):
        sess.run(m.optimizer, feed_dict={m.environment_: inputs})
        #if np.sqrt(epoch+1)%1== 0:
        if epoch%args.save_every==0:
            c = sess.run([ mvalid.status], feed_dict={mvalid.environment_: inputs})
            c = np.mean(c)
            print("Epoch:", '%04d' % (epoch+1), "cost=",c, )
            checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step =epoch)

if __name__=="__main__":
    main_example()
    
    

    
#[1]: https://github.com/sherjilozair/char-rnn-tensorflow/blob/master/train.py