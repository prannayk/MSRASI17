from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math
import time
import os
import sys

initializer_main = tf.random_normal_initializer(stddev=0.02)

class ConvSiamese():
	def __init__(self, embedding_size):
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
	def lstm(self, word_embed, scope. flag=True, reverse=True):
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
	def attention_over_sequence(self, word_embed, flag=True):
		h1 = tf.layers.dense(word_embed, units=self.embedding_size // 2, 
			activation=tf.tanh, kernel_initializer=self.initializer, 
			use_bias=True,
			name="dense_1", reuse=scope.reuse)
		h1_norm = self.normalize(h1)
		h2 = tf.layers.dense(h1_norm, units=1, activation=None,
			kernel_initializer=self.initializer, use_bias=True, 
			name="dense_2", reuse=scope.reuse)
		h2_reshape = tf.reshape(h2, shape=[self.batch_size, self.word_max_len])
