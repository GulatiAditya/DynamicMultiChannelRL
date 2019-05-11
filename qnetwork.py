"""
The main crux of project qnetwork,environment and experience memory
Code Author - Aditya Gulati and Abhinav Gupta
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np 
import tensorflow as tf 
import pandas as pd 
from collections import deque

tf.keras.backend.clear_session()  # For easy reset of notebook state.

from tensorflow.keras import layers

# Initial Environment
class channel_env:
	def __init__(self,n_channels):
		self.n_channels = n_channels
		self.reward = 1

		self.action_set = np.arange(n_channels)
		self.action = -1
		self.observation = -1

# QNetwork as given in paper
class QNetwork:
	def __init__(self,learning_rate,state_size,action_size,hidden_size,name="Channel_QNetwork"):

		with tf.variable_scope("Channel_QNetwork"):
			self.input_in =  tf.placeholder(tf.float32, [None,state_size],name="Input")
			self.action = tf.placeholder(tf.int32,[None],name="action")
			# print(type(self.action))
			action_onehot_vec = tf.one_hot(self.action,action_size)

			self.semiGTq = tf.placeholder(tf.float32,[None],name="actuals_Q")

			self.w1 = tf.Variable(tf.random_uniform([state_size,hidden_size]))
			self.b1 = tf.Variable(tf.constant(0.01,shape=[hidden_size]))
			self.h1 = tf.matmul(self.input_in,self.w1) + self.b1 
			self.h1 = tf.nn.relu(self.h1)

			self.w2 = tf.Variable(tf.random_uniform([hidden_size,hidden_size]))
			self.b2 = tf.Variable(tf.constant(0.01,shape=[hidden_size]))
			self.h2 = tf.matmul(self.h1,self.w2) + self.b2
			self.h2 = tf.nn.relu(self.h2)

			self.w_outlayer = tf.Variable(tf.random_uniform([hidden_size,action_size]))
			self.b_outlayer = tf.Variable(tf.random_uniform([action_size]))
			self.out_layer = tf.matmul(self.h2,self.w_outlayer) + self.b_outlayer

			self.Q_pred = tf.reduce_sum(tf.multiply(self.out_layer, action_onehot_vec), axis=1)

			self.Q_loss = tf.reduce_mean(tf.square(self.semiGTq - self.Q_pred))
			self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.Q_loss)


# Experience memory of fixed max buffer size
class ExpMemory():
    def __init__(self, in_size):
        self.buffer_in = deque(maxlen=in_size)
    def add(self, exp):
        self.buffer_in.append(exp)
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer_in)),size=batch_size, replace=False)
        res = []
        for i in idx:
            res.append(self.buffer_in[i])
        return res    
