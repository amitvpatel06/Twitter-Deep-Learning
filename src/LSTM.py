import getpass
import sys
import time

import numpy as np
from copy import deepcopy

from utils.batch_feeder import *
from utils.parse_data import *
from config import Config
from tensorflow.python.ops.seq2seq import sequence_loss

import tensorflow as tf

class RNN: 

	def __init__(self, config, debug):
		self.config = config
		self.load_data(debugMode=debug)
		predictions = self.build_model_graph()
		self.add_training_objective(predictions)


	def build_model_graph(self): 
		# create inputs placeholders for inputs
		self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.steps), name = 'inputs')
		self.labels_placeholder = tf.placeholder(tf.float32, shape=(None, 2), name ='labels')
		self.dropout_placeholder = tf.placeholder(tf.float32, name = 'dropout')
		self.L = tf.get_variable('L', shape = [len(self.vocab.word_to_index), self.config.embed_size],
									initializer=tf.contrib.layers.xavier_initializer())
		with tf.device('/cpu:0'):
			embed = tf.nn.embedding_lookup(self.L, self.input_placeholder)
			inputs = [tf.squeeze(x, [1]) for x in tf.split(1, self.config.steps, embed)]

		lstm_forward = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size)
		lstm_backward = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size)
		forward = tf.nn.rnn_cell.MultiRNNCell([lstm_forward] * 3)
		backward = tf.nn.rnn_cell.MultiRNNCell([lstm_backward] * 3)


		rnn_outputs, f_final, b_final = tf.nn.bidirectional_rnn(forward, backward, inputs, dtype=tf.float32)

		self.final_state = rnn_outputs[-1]
		with tf.variable_scope('projection'):
			U = tf.get_variable('U', shape=[2 * self.config.hidden_size, 2])
			b_2 = tf.get_variable('b2', shape=[2])
			outputs = (tf.matmul(self.final_state, U) + b_2)
			if self.config.l2:
				tf.add_to_collection('L2', self.config.l2 * (tf.nn.l2_loss(U)))
				tf.add_to_collection('L2', self.config.l2 * (tf.nn.l2_loss(b_2)))
		predictions = tf.cast(outputs, 'float32')
		return predictions

	def add_training_objective(self, predictions):
		correct = tf.equal(tf.argmax(tf.nn.softmax(predictions),1), tf.argmax(self.labels_placeholder,1))
		self.percent = tf.reduce_mean(tf.cast(correct,tf.float32))
		self.loss = tf.reduce_sum(
			tf.nn.softmax_cross_entropy_with_logits(predictions, self.labels_placeholder)) + tf.add_n(tf.get_collection('L2'))	
		optimizer = tf.train.AdamOptimizer(self.config.lr)
		self.train_grad = optimizer.minimize(self.loss)

	def load_data(self, debugMode=False):
		self.vocab = Vocab()
		if not debugMode:
			self.encoded_train, self.labels = create_data_set(self.vocab, 'test.csv', 
				steps=self.config.steps)
			self.encoded_valid, self.valid_labels = create_data_set(self.vocab, 'dev.csv', 
				steps=self.config.steps)
		else: 
			self.encoded_train, self.labels = create_data_set(self.vocab, "Test.csv")

	def run_epoch(self, session, data, train=None, print_freq=10):
		if data == "train" or data == 'debug':
			encoded = self.encoded_train
			labels = self.labels
		elif data == "valid":
			encoded = self.encoded_valid
			labels = self.valid_labels
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
				self.dropout_placeholder: dropout,
			}
			loss, state, percent, _ = session.run(
				[self.loss, self.final_state, self.percent, self.train_grad], feed_dict=feed)
			total_percent.append(percent * 100)
			total_loss.append(loss)
			if step % print_freq == 0:
				print ('\r{} / {} ,{}% : CE = {}'.format(
					step, total_steps, np.mean(total_percent), np.mean(total_loss)))
		return (np.mean(total_loss), np.mean(total_percent))


def run_RNN(num_epochs, debug=False):
	config = Config()
	
	with tf.variable_scope('RNN') as scope:
		model = RNN(config, debug)
	init = tf.initialize_all_variables()
	saver = tf.train.Saver()
	with tf.Session() as session:
		session.run(init)
		best_val_ce = float('inf')
		best_val_epoch = 0
		for epoch in xrange(num_epochs):
			print 'Epoch {}'.format(epoch)
			start = time.time()
			train_ce, train_perecent = model.run_epoch(
				session, 'debug',
				train=model.train_grad)
			if not debug: 
				valid_ce, valid_percent = model.run_epoch(session, 'valid')
				print 'Validation CE loss: {}'.format(valid_ce)
				if valid_ce < best_val_ce:
					best_val_pp = valid_ce
					best_val_epoch = epoch
					saver.save(session, './lstm.weights')
				if epoch - best_val_epoch > config.early_stopping:
					break
				epoch_summary = {
					'Epoch': epoch,
					'Train CE': train_ce,
					'Valid CE': valid_ce,
					'Train Percent': train_percent,
					'Valid Percent': valid_percent
				}
				summary.append(epoch_summary)
			else:
				epoch_summary = {
					'Epoch': epoch,
					'Train CE': 1,
					'Valid CE': 1,
					'Train Percent': 1,
					'Valid Percent': 1
				}
				summary.append(epoch_summary)
		print summary
		write_summary(summary, ['Epoch', 'Train CE', 'Valid CE',
											'Train Percent', 'Valid Percent'], 'summary.csv')
		print 'Total time: {}'.format(time.time() - start)

if __name__ == "__main__":
	run_RNN(30, debug=True)