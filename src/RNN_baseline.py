import getpass
import sys
import time

import numpy as np
from copy import deepcopy

from batch_feeder import *
from parse_data import *
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
		# creat inputs placeholders for inputs
		self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.steps), name = 'inputs')
		self.labels_placeholder = tf.placeholder(tf.float32, shape=(None, 2), name ='labels')
		self.dropout_placeholder = tf.placeholder(tf.float32, name = 'dropout')

		with tf.device('/cpu:0'):
			embed = tf.nn.embedding_lookup(self.L, self.input_placeholder)
			inputs = [tf.squeeze(x, [1]) for x in tf.split(1, self.config.steps, embed)]

		rnn_outputs = []
		with tf.variable_scope('RNN_Layer') as scope:
			for i in range(len(inputs)):
				if i > 0 : 
					scope.reuse_variables()
				else: 
					self.state = tf.zeros([self.config.batch_size, self.config.hidden_size])
				H = tf.get_variable('H', shape = [self.config.hidden_size, self.config.hidden_size]) 
				I = tf.get_variable('I', shape = [self.config.embed_size, self.config.hidden_size]) 
				b = tf.get_variable('b', shape = [self.config.hidden_size,])
				self.state = tf.nn.sigmoid(
					tf.matmul(self.state, H) + tf.matmul(inputs[i], I) + b)
				rnn_outputs.append(tf.nn.dropout(self.state, self.dropout_placeholder))
			self.final_state = rnn_outputs[-1]
		with tf.variable_scope('projection'):
			U = tf.get_variable('U', shape=[self.config.hidden_size, 2])
			b_2 = tf.get_variable('b2', shape=[1])
			outputs = (tf.matmul(self.final_state, U) + b_2)
		predictions = tf.cast(outputs, 'float32')
		return predictions

	def add_training_objective(self, predictions):
		self.predictions = tf.nn.softmax(predictions)
		self.loss = tf.reduce_sum(
			tf.nn.softmax_cross_entropy_with_logits(predictions, self.labels_placeholder))
		optimizer = tf.train.AdamOptimizer(self.config.lr)
		self.train_grad = optimizer.minimize(self.loss)
		# need to add training objectives and feed dict code

	def load_data(self, debugMode=False):
		self.vocab = Vocab()
		if not debugMode:
			self.encoded_train, self.labels, self.L = create_data_set(self.vocab, '../data/train.csv', 
				'../data/word vectors/glove.twitter.27B.50d.txt', steps=self.config.steps)
			self.encoded_valid, self.valid_labels, _ = create_data_set(self.vocab, '../data/valid.csv', 
				'../data/word vectors/glove.twitter.27B.50d.txt', steps=self.config.steps)
			self.encoded_test, self.valid_test, _ = create_data_set(self.vocab, '../data/test.csv', 
				'../data/word vectors/glove.twitter.27B.50d.txt', steps=self.config.steps)
		else: 
			self.encoded_train, self.labels, self.L = create_data_set(self.vocab, "Test.csv", "Test_vecs.txt", steps=10)

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
			loss, state, predictions,_ = session.run(
				[self.loss, self.final_state, self.predictions, self.train_grad], feed_dict=feed)
			percent = percent_right(predictions, labels, self.config.batch_size) * 100
			total_percent.append(percent)
			total_loss.append(loss)
			if step % print_freq == 0:
				sys.stdout.write('\r{} / {} ,{}% : CE = {}'.format(
					step, total_steps, np.mean(total_percent), np.mean(total_loss)))
				sys.stdout.flush()
		return np.mean(total_loss)
 
def percent_right(predicted, actual, number):
	right = 0
	for prob, actual in zip(predicted, actual):
		if(prob[0] > 0.5):
			if actual[0] == 1:
				right +=1
		else:
			if actual[0] == 0:
				right +=1
	return (right * 1.0 /number)

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
			train_ce = model.run_epoch(
				session, 'debug',
				train=model.train_grad)
			#valid_ce = model.run_epoch(session, 'valid')
			print 'Training CE loss: {}'.format(train_ce)
			print 'Validation CE loss: {}'.format(valid_ce)
			if valid_ce < best_val_ce:
				best_val_pp = valid_ce
				best_val_epoch = epoch
				saver.save(session, './rnn.weights')
			if epoch - best_val_epoch > config.early_stopping:
				break
		print 'Total time: {}'.format(time.time() - start)

if __name__ == "__main__":
	run_RNN(10, debug=False)