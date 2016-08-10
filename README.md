#Reinforcement Learning

We create a generic reinforcement learning module. We leave configuration of data types adjustable and the policy and fitness function are left blank and adjustable for the user to adjust to their liking. 

## Installation and configuration

```git clone https://github.com/LiamConnell/ReinforcementLearning_Lorenzo.git```

I recomend the [anaconda](https://www.continuum.io/downloads) distribution of python. It includes difficult to install packages like numpy and pandas and has installation instructions for both linux and windows on the website. 

Tensorflow installation should be simple as well with anaconda. [Instructions](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#anaconda-installation) are provided and I recomend using the pip instructions that they provide. Make sure to update pip before attempting installation with `pip install --upgrade pip`.

##Documentation

Execution by running `python reinforcement.py` with any arguments tagged afterwards. See reinforcemnt.py for list of arguments with descriptions. Everything from importing data to saving/reinitializing the model to configuring the model shape and size can be configured in this manner. 

For example if we want to train a model that has data in 150 dimensions and the data is stored in MYdata.csv. We can call.

```
    python reinforcement.py --data_file 'MYdata.csv' --data_size 150
```


###Loading the model
We load the model, stored in `reinforcement_lstm_model.py`. 
```
    sess = tf.InteractiveSession()
    # initialize variables to random values
    with tf.variable_scope("model", reuse=None):
        m = Model(config = SmallConfig)
    with tf.variable_scope("model", reuse=True):
        mvalid = Model(config = SmallConfig, training=False)
    sess.run(tf.initialize_all_variables())
```
`m` is a class representing the model that will train the LSTM and `mvalid` is used for validation. The distinction is important because we can add dropout and other tools to improve our network that require activation for training and deactivation for validation. More models can be added with other `num_steps` but other configuration variables should stay the same.

###Train model

```
    for epoch in range(args.num_epochs):
        sess.run(m.optimizer, feed_dict={m.environment_: inputs})
        #if np.sqrt(epoch+1)%1== 0:
        if epoch%args.save_every==0:
            c = sess.run([ mvalid.status], feed_dict={mvalid.environment_: inputs})
            c = np.mean(c)
            print("Epoch:", '%04d' % (epoch+1), "cost=",c, )
            checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step =epoch)
```

We run the optimizer many times using the `m` model. Frequently, we evaluate the cost function for out model to make sure that it is descending. We can evaluate other variables as well such as `reward`. In a real scenario, data should be separated into training and evaluation sections. Stochastic gradient descent should also be considered by batching the testing data if data sizes are very large. 
