import getpass
import sys
import time

import numpy as np
from copy import deepcopy

from utils.batch_feeder import *
from utils.parse_data import *
from config import Config

import tensorflow as tf

class RNN: 

	def __init__(self):
		self.config = Config()
		self.vocab = Vocab()	

	def build_model_graph(self): 
		# creat inputs placeholders for inputs
		self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.num_steps), name = 'inputs')
		self.labels_placeholder = tf.placeholder(tf.float32, shape=(None, 2), name ='labels')
		self.dropout_placeholder = tf.placeholder(tf.float32, name = 'dropout')
		self.L =  tf.placeholder(tf.float32,shape = [len(self.vocab), self.config.embed_size], name = 'L')
		self.keep_prob = tf.placeholder(tf.float32)

		with tf.device('/cpu:0'):
			embed = tf.nn.embedding_lookup(L, input_placeholder)
			inputs = [tf.squeeze(x, [1]) for x in tf.split(1, self.congif.num_steps, embed)]

		rnn_outputs = []
		with tf.variable_scope('RNN_Layer') as scope:
			for i in range(len(inputs)):
				if i > 0 : 
					scope.reuse_variables()
				else: 
					self.initial_state = tf.zeros([self.config.batch_size, self.hidden_size])
					state = tf.zeros([self.config.batch_size, self.hidden_size])
				H = tf.get_variable('H', shape = [self.config.hidden_size, self.config.hidden_size]) 
				I = tf.get_variable('I', shape = [self.config.embed_size, self.config.hidden_size]) 
				b = tf.get_variable('b', shape = [hidden_size,])
				state = tf.nn.sigmoid(
					tf.matmul(self.state, H) + tf.matmul(inputs[i], I) + b)
				rnn_outputs.append(tf.nn.dropout(state, self.keep_prob))
			self.final_state = rnn_outputs[-1]


		with tf.variable_scope('projection'):
			U = tf.get_variable('U', shape=[hidden_size, len(self.vocab)])
			b_2 = tf.get_variable('b2', shape=[len(self.vocab)])
			outputs = [(tf.matmul(x, U) + b_2) for x in rnn_outputs]

		predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]


		# need ot add training objectives and feed dict code
