#Reinforcement Learning

We create a generic reinforcement learning module. We leave configuration of data types adjustable and the policy and fitness function are left blank and adjustable for the user to adjust to their liking. 

## Installation and configuration

```git clone https://github.com/LiamConnell/ReinforcementLearning_Lorenzo.git```

I recomend the [anaconda](https://www.continuum.io/downloads) distribution of python. It includes difficult to install packages like numpy and pandas and has installation instructions for both linux and windows on the website. 

Tensorflow installation should be simple as well with anaconda. [Instructions](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#anaconda-installation) are provided and I recomend using the pip instructions that they provide. Make sure to update pip before attempting installation with `pip install --upgrade pip`.

##Documentation

Execution by running `python reinforcement.py`.

###Configuration
A configuration class contains variables that will be used to configure the model. 

```
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
 ```
 
* `input_size` is the dimension of each step of data
* `output_size` is the dimension of what the LSTM should output. For instance, if we are chosing an integer between -3 and 3, there will be seven options, so the `output_size` is 7.
* `num_steps` this is how many steps will come through the LSTM at a time.
* `num_samples`: we use multi-sampling in order to aid in out stochastic descent. This is a magic number and must be chosen experimentally.
* `learning_rate`: rate of descent, chosen experimentally. When in doubt, chose smaller learning rate.
* `hidden_side`: the size of the LSTM state
* `num_layers`: the depth of the LSTM
* `forget_bias`: the rate that our LSTM forgets the past

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
    for epoch in range(200):
        sess.run(m.optimizer, feed_dict={m.environment_: inputs})
        if np.sqrt(epoch+1)%1== 0:
            c = sess.run([ mvalid.cost], feed_dict={mvalid.environment_: inputs})
            c = np.mean(c)
            print("Epoch:", '%04d' % (epoch+1), "cost=",c, )
```

We run the optimizer many times using the `m` model. Frequently, we evaluate the cost function for out model to make sure that it is descending. We can evaluate other variables as well such as `reward`. In a real scenario, data should be separated into training and evaluation sections. Stochastic gradient descent should also be considered by batching the testing data if data sizes are very large. 

## TODO
* Add methods for saving and recovering trained models
* What is the purpose of `status`?
* Not computationally efficient to check `status` each timestep and abandon if below a threshold as described in project description.

