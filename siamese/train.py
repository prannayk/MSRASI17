from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math
import time
import os
import sys
sys.path.append( '../util/')
from generators import *
from loader import *
from similar_tokens import *
from argument_loader import *

dataset, query_type, filename, num_steps, num_steps_roll, num_steps_train, expand_flag,lr_, matchname = import_arguments(sys.argv)

char_batch_dict, word_batch_dict,data, count, dictionary, reverse_dictionary, word_max_len, char_max_len, vocabulary_size, char_dictionary, reverse_char_dictionary, data_index, char_data_index, buffer_index, batch_list, char_batch_list, word_batch_list, char_data = build_everything(dataset)

class ConvSiamese():
	def __init__(self, embedding_size=128, batch_size=64, word_max_len=word_max_len,
		learning_rate = 0.01, vocabulary_size=vocabulary_size):
		self.embedding_size = embedding_size
		self.initializer = tf.random_normal_initializer(stddev=0.02)
	def normalize(self, X, reuse=False, name=None, flag=False):
		if not flag : 
			mean , vari = tf.nn.moments(X, 0, keep_dims =True)
		else:
			mean, vari = tf.nn.moments(X, [0,1,2], keep_dims=True)
		return tf.nn.batch_normalization(X, mean, vari, offset=None, 
			scale=None, variance_epsilon=1e-6, name=name)
	def convolve_word(self, word_embed, scope):
		word_embed_reshape = tf.reshape(word_embed, shape=[self.batch_size
			self.word_max_len, 1, self.embedding_size])
		h1 = tf.layers.conv2d(word_embed_reshape, filters=128, 
			kernel_size=[4,1]. strides=[1,1], padding='SAME',
			activation=tf.tanh, kernel_initializer=self.initializer,
			name="conv_1")
		h1_max_pool = tf.layers.max_pooling2d(h1, pool_size=[3,1],
			strides=[2,1],padding='SAME',name="max_pool_1")
		h1_norm = self.normalize(h1_max_pool)
		h2 = tf.layers.conv2d(h1_norm, filters=64, 
			kernel_size=[4,1]. strides=[1,1], padding='SAME',
			activation=tf.tanh, kernel_initializer=self.initializer,
			name="conv_2")
		h2_max_pool = tf.layers.max_pooling2d(h2, pool_size=[3,1], 
			strides=[2,1], padding='SAME', name="max_pool_2")
		h2_norm = self.normalize(h2_max_pool)
		h2_reshape = tf.reshape(h2_norm, shape=[self.batch_size, self.word_max_len, 64])
		return h2_reshape
	def lstm(self, word_embed, scope, flag=True, reverse=True):
		batch_size = int(word_embed.get_shape()[0])
		seq_len = int(word_embed.get_shape()[1])
		lstm = tf.contrib.rnn.BasicLSTMCell(64, reuse=scope.reuse)
		state = lstm.zero_state(batch_size, dtype=tf.float32)
		repeat_flag = False
		for i in range(seq_len):
			if flag or repeat_flag :
				scope.reuse_variables()
			repeat_flag = True
			if not reverse :
				cell_output, state = lstm(word_embed[:,l,:])
				output_state = tf.reshape(cell_output, shape=[batch_size, 1, 64])
				if repeat_flag : 
					output = tf.concat([output,output_state],axis=1)
				else:
					output = output_state
			else : 
				cell_output, state = lstm(word_embed[:,seq_len -l - 1 ,:])
				output_state = tf.reshape(cell_output, shape=[batch_size, 1, 64])
				if repeat_flag : 
					output = tf.concat([output_state,output],axis=1)
				else:
					output = output_state
		return output
	def bilstm(self, word_embed, scope):
		with tf.variable_scope("forward_lstm") as scope:
			output_forward = self.lstm(word_embed, reverse=False)
		with tf.variable_scope("backward_lstm") as scope:
			output_backward = self.lstm(word_embed, reverse=True)
		return tf.concat([output_forward,output_backward], axis=2)
	def attention_over_sequence(self, word_embed):
		h1 = tf.layers.dense(word_embed, units=self.embedding_size // 2, 
			activation=tf.tanh, kernel_initializer=self.initializer, 
			use_bias=True,
			name="dense_1", reuse=scope.reuse)
		h1_norm = self.normalize(h1)
		h2 = tf.layers.dense(h1_norm, units=1, activation=None,
			kernel_initializer=self.initializer, use_bias=True, 
			name="dense_2", reuse=scope.reuse)
		h2_reshape = tf.reshape(h2, shape=[self.batch_size, self.word_max_len])
		h2_softmax = tf.nn.softmax(h2_reshape)
		return h2_softmax
	def energy(self, embedding, scope, energy_type="cosine"):
		energy_type = energy_type.lower()
		norm1 = tf.norm(embedding[0], axis=1)
		norm2 = tf.norm(embedding[1], axis=1)
		embedding1_norm = embedding[0] / norm1
		embedding2_norm = embedding[1] / norm2
		if energy_type == "cosine":
			energy_cosine = tf.reduce_sum(embedding1_norm*embedding2_norm)
			return energy_cosine
		elif energy_type == "l2_loss":
			energy_l2 = tf.nn.l2_loss(embedding1_norm - embedding2_norm)
			return energy_l2
		else :
			raise NotImplemented
	def architecture_lstm(self, placeholders, markers):
		word_embedding = []
		lstm_embedding = []
		with tf.variable_scope("word_embedding"):
			word_embedding = tf.variable_scope(tf.random_normal(stddev=0.02, 
				shape=[self.vocabulary_size, self.word_embedding_size]))
			norm = tf.norm(word_embedding, axis=1)
			norm_embedding = word_embedding / norm
			word_embedding.append(tf.nn.embedding_lookup(norm_embedding, placeholders[0]))
			word_embedding.append(tf.nn.embedding_lookup(norm_embedding, placeholders[1]))
		with tf.variable_scope("bilstm") as scope:
			lstm_embedding.append(self.bilstm(word_embedding[0], scope))
			lstm_embedding.append(self.bilstm(word_embedding[1], scope))
		with tf.variable_scope("energy") as scope:
			energy = self.energy(lstm_embedding, scope, energy_type="cosine")
		loss = tf.reduce_mean(tf.square(energy - markers))
		return loss

		return cross_entropy_loss
	def build_model(self):
		tweet = []
		tweet.append(tf.placeholders(tf.int32, shape=[self.batch_size, self.word_max_len]))
		tweet.append(tf.placeholders(tf.int32, shape=[self.batch_size, self.word_max_len]))
		markers = tf.placeholders(tf.float32, shape=[self.batch_size])
		loss = -self.architecture_lstm(tweet)

		optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

		return tweet, markers, loss, optimizer

data_index, batch, labels = generate_batch(data, data_index, batch_size=8, num_skips=2, skip_window=1,)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

lambda_1, tweet_batch_size, expand_start_count, query_name, query_tokens, query_tokens_alternate, char_batch_size, num_sampled, valid_examples, valid_window, valid_size, skip_window, num_skips, embedding_size, char_vocabulary_size, batch_size, num_char_skips, skip_char_window = setup(char_dictionary, dictionary, query_type)

architecture = ConvSiamese(learning_rate=lr_)
tweet, loss, optimizer = architecture.build_model()

epoch = 100
batch_size = 32
total_tweets = len(word_batch_dict) - (len(word_batch_dict) % batch_size)
print_interval = 10

for ep in range(epoch):
	start_time = time.time()
	average_val = 0
	assert total_tweets % batch_size == 0
	for i in range(total_tweets // batch_size) : 
		tweet_value, marker_value = generate_pair()
		feed_dict = {
			tweet : tweet_value
			markers : marker_value
		}
		_, loss_val = session.run([optimizer, loss], feed_dict = feed_dict)
		average_val += (loss_val / print_interval)
		if i % print_interval == 0:
			print("Batches : %d ,Loss: %.2f and time taken %.2f"%(i*batch_size, average_val, time.time()-start_time))
			average_val = 0
			start_time = time.time()
	architecture.create_qrel(word_batch_dict, char_batch_dict)
