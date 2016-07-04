import getpass
import sys
import time

import numpy as np
from copy import deepcopy

from utils.batch_feeder import *
from utils.parse_data import *
from tensorflow.python.ops.seq2seq import sequence_loss
from config import Config
import tensorflow as tf


class Filters:
	def __init__(self):
		self.filter_sizes =  [3, 4, 5]
		self.num_filters = 40

class CNN: 

	def __init__(self, config, filters, debug):
		self.config = config
		self.filters = filters
		self.load_data(debugMode=debug)
		predictions = self.build_model_graph()
		self.inference = tf.nn.softmax(predictions)
		self.add_training_objective(predictions)


	def build_model_graph(self): 
		# create inputs placeholders for inputs
		self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.steps), name = 'inputs')
		self.labels_placeholder = tf.placeholder(tf.float32, shape=(None, 2), name ='labels')
		self.dropout_placeholder = tf.placeholder(tf.float32, name = 'dropout')
		self.L = tf.get_variable('L', shape = [len(self.vocab.word_to_index), self.config.embed_size],
									initializer=tf.contrib.layers.xavier_initializer())

		with tf.device('/cpu:0'):
			embed_inputs = tf.nn.embedding_lookup(self.L, self.input_placeholder)
			# add a dim for tf.nn.conv2d
			self.expanded_inputs = tf.expand_dims(embed_inputs, -1)


		pool_outputs = []
		for i, size in enumerate(self.filters.filter_sizes):
			with tf.name_scope("convolution_size_{}".format(size)):
				shape = [size, self.config.embed_size, 1, self.filters.num_filters]
				filt = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="filter")
				convolved_features = tf.nn.conv2d(
						self.expanded_inputs,
						filt,
						strides= [1,1,1,1],
						padding="VALID",
						name="conv_features"
				)
				bias = tf.Variable(tf.constant(0.1, shape=[self.filters.num_filters]), name="bias")
				if self.config.l2:
					tf.add_to_collection('L2', self.config.l2 * (tf.nn.l2_loss(filt)))
					tf.add_to_collection('L2', self.config.l2 * (tf.nn.l2_loss(bias)))
				conv_hidden = tf.nn.relu(tf.nn.bias_add(convolved_features, bias), name="RELU")
				pooled_features = tf.nn.max_pool(conv_hidden, 
												ksize=[1, self.config.steps - size + 1, 1, 1],
												strides=[1,1,1,1],
												padding='VALID',
												name="pooled_features")
				pool_outputs.append(pooled_features)

		self.hidden = tf.reshape(tf.concat(3, pool_outputs), [-1, self.filters.num_filters * len(self.filters.filter_sizes)])
		self.hidden_with_dropout = tf.nn.dropout(self.hidden, self.dropout_placeholder)
		# embed into classification vector
		with tf.variable_scope('projection'):
			U = tf.get_variable('U', shape=[self.filters.num_filters * len(self.filters.filter_sizes), 2])
			b_2 = tf.get_variable('b2', shape=[2])
			outputs = (tf.matmul(self.hidden_with_dropout, U) + b_2)
			if self.config.l2:
				tf.add_to_collection('L2', self.config.l2 * (tf.nn.l2_loss(U)))
				tf.add_to_collection('L2', self.config.l2 * (tf.nn.l2_loss(b_2)))
		return outputs


	def add_training_objective(self, predictions):
		correct = tf.equal(tf.argmax(tf.nn.softmax(predictions),1), tf.argmax(self.labels_placeholder,1))
		self.percent = tf.reduce_mean(tf.cast(correct,tf.float32))
		self.CE = tf.reduce_sum(
			tf.nn.softmax_cross_entropy_with_logits(predictions, self.labels_placeholder))
		self.loss = tf.add_n(tf.get_collection('L2')) + self.CE
		optimizer = tf.train.AdamOptimizer(self.config.lr)
		self.train_grad = optimizer.minimize(self.loss)
		# need to add training objectives and feed dict code

	def load_data(self, debugMode=False):
		self.vocab = Vocab()
		if not debugMode:
			self.encoded_train, self.labels = create_data_set(self.vocab, 'dev.csv', 
				steps=self.config.steps)
			self.encoded_valid, self.valid_labels = create_data_set(self.vocab, 'test.csv', 
				steps=self.config.steps)
			self.encoded_test, self.valid_test = create_data_set(self.vocab, 'test.csv', 
				steps=self.config.steps)
		else: 
			self.encoded_train, self.labels = create_data_set(self.vocab, "Test.csv", steps=self.config.steps)
		
	def run_epoch(self, session, data, train=None, print_freq=100):
		if data == "train" or data == 'debug':
			encoded = self.encoded_train
			labels = self.labels
		elif data == "valid":
			encoded = self.encoded_valid
			labels = self.valid_labels
		else: 
			encoded = self.encoded_test
			labels = self.valid_test
		dropout = self.config.dropout
		if not train:
			train = tf.no_op()
			dropout = 1
		total_loss = []
		total_percent = []
		total_steps = sum(1 for x in data_iterator(encoded, labels, batch_size=self.config.batch_size))
		for step, (batch_inputs, batch_labels) in enumerate(data_iterator(encoded, labels, batch_size=self.config.batch_size)):
			feed = {
				self.input_placeholder: batch_inputs,
				self.labels_placeholder: batch_labels,
				self.dropout_placeholder: dropout
			}
			loss, predictions, percent, _ = session.run(
				[self.CE, self.inference, 
					self.percent, self.train_grad], feed_dict=feed)
			total_percent.append(percent*100)
			total_loss.append(loss)
			if step % print_freq == 0:
				sys.stdout.write('\r{} / {} ,{}% : CE = {}'.format(
					step, total_steps, np.mean(total_percent), np.mean(total_loss)))
				sys.stdout.flush()
		return np.mean(total_loss)

def run_CNN(num_epochs, debug=False):
	config = Config()
	filters = Filters()
	with tf.variable_scope('CNN') as scope:
		model = CNN(config, filters, debug)
	init = tf.initialize_all_variables()
	saver = tf.train.Saver()
	with tf.Session() as session:
		session.run(init)
		best_val_ce = float('inf')
		best_val_epoch = 0
		for epoch in xrange(num_epochs):
			print 'Epoch {}'.format(epoch)
			start = time.time()
			train_ce = model.run_epoch(
				session, 'debug',
				train=model.train_grad)
			print 'Training CE loss: {}'.format(train_ce)
			if not debug:
				valid_ce = model.run_epoch(session, 'valid')
				print 'Validation CE loss: {}'.format(valid_ce)
				if valid_ce < best_val_ce:
					best_val_pp = valid_ce
					best_val_epoch = epoch
					saver.save(session, './cnn.weights')
				if epoch - best_val_epoch > config.early_stopping:
					break
		print 'Total time: {}'.format(time.time() - start)

if __name__ == "__main__":
	run_CNN(10, debug=False)