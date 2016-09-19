# Reinforcement Learning

We create a generic reinforcement learning module. We leave configuration of data types adjustable and the policy and fitness function are left blank and adjustable for the user to adjust to their liking. 

## Installation and configuration

```
git clone https://github.com/LiamConnell/ReinforcementLearning_Lorenzo.git
```

I recomend the [anaconda](https://www.continuum.io/downloads) distribution of python. It includes difficult to install packages like numpy and pandas and has installation instructions for both linux and windows on the website. 

Tensorflow installation should be simple as well with anaconda. [Instructions](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#anaconda-installation) are provided and I recomend using the pip instructions that they provide. Make sure to update pip before attempting installation with `pip install --upgrade pip`.

Once everything is installed, you can run the script through a toy example by running `python reinforcement.py` to make sure everything is working.

## Documentation

To execute we will run `python reinforcement.py` in the terminal. Arguments to customize the model can be added as "tags". For example if we want to train a model that has data in 150 dimensions and the data is stored in MYdata.csv. We can call:

```
python reinforcement.py --data_file 'MYdata.csv' --input_size 150
```

Here is a list of all the arguments that we can use like this:

* `--data_file` where we load the data from (a .csv file)
* `--save_dir` the directory that we want to save our trained models
* `--init_from` if continuing to train an already saved model, the path to that model
* `--input_size` size of input vector for each time step
* `--output_size` dimension of the vector that the LSTM should output each timestep
* `--num_steps` the number of steps that go into the LSTM at a time
* `--num_samples` the number of samples to make during training
* `--learning rate` the learning rate that the gradient descent will progress at
* `--hidden_size` the size of the LSTM hidden state
* `--num_layers` the number of layers of the LSTM (see tensorflow: MultiRNNCell)
* `--forget_bias` the forget bias fo the LSTM (how quickly it forgets old information to make room for new)
* `--num_epochs` the number of epochs that training will go through (how long to train for)
* `--save_every` the frequency that models should be saved (every 'N' epochs)

Aside from the tuning of these variables, it might be necessary to modify the `read_data_fn` on line 62 of `reinforcement.py` if your data is not in csv form. I refer the reader to the pandas documentation on importing data. Pandas makes it simple manage a small dataflow like this and you can almost always find a one-liner that imports your data to the correct format. 

The nature of deep learning is that creating the correct model is only half the battle. A typical user of this module will run `reinforcement.py` and with the correct tags to describe the data, but will still find that the cost function doesnt descend or even increases. This is a common problem with deep learning: the parameters are very hard to generalize between models and often require significant experimentation in order to find what works. This is where coursera and udacity courses work well. 

### STATUS and REWARD functions

The status and reward functions are necessary in order to define the "game" and goal that the deep network should learn. The two functions (methods technically) can be found in line 77 and 87 of `reinforcement_lstm_model.py` respectively. `get_status` currently has a dummy function that will almost certainly have to be changed, while `get_reward` has a pretty typical function that will likely not require modification for most problems. 

When changing the `get_status` method, keep in mind the following. The self.status variable should return what can be seen as the "score" of the game or problem that is being solved. It will almost certainly use either the `self.training_target_cols` variable which represents the output values of the lstm that correspond to the sampled action that the model "decided" to make or `self.sample` which represents the actual action. Outside data (target values, etc.) can be passed in through the `feed_dict` on line 115 and 118 of `reinforcement.py`. Since it is rare to pass in target data when doing reinforcement learning (in fact it is not actually reinforcement learning if there is target data), I have not programmed that functionality explicitly, but it is possible with only minor code adjustments. 


## What is happening inside the model?
There are a few concepts that might help the reader if they are explained explicitly. The nature of tensorflow, being a tensor-based calculator, means that we define the parts of the program in a different order than many other python programs or SQL scripts. It isnt necessary to understand this in order to run the program, but as the user tries to fine-tune the model, it might help to know what is going on. It is a pretty typical organization for a tensorflow module.

### Loading the model
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

### Train model

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
