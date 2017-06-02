import tensorflow as tf
import numpy as np
import json
import re
import math
import time

class tweet2vec():
	def __init__(batch_size, vocabulary_size, word_max_len,word_embedding_size=128, tweet_embedding_size=256, num_classes=3):
		self.batch_size = batch_size
		self.word_embedding_size = word_embedding_size
		self.num_classes = num_classes
		self.tweet_embedding_size = tweet_embedding_size
		self.vocabulary_size = vocabulary_size
		self.word_max_len = word_max_len

		self.grum_weights = tf.stack([tf.Variable(tf.random_normal(stddev=1e2/math.sqrt(self.word_embedding_size*self.tweet_embedding_size),shape=[self.word_embedding_size,self.tweet_embedding_size]))]*self.batch_size)
		self.gru1_weights = tf.stack([tf.Variable(tf.random_normal(stddev=1e2/math.sqrt(self.word_embedding_size*self.tweet_embedding_size),shape=[self.word_embedding_size,self.tweet_embedding_size]))]*self.batch_size)

		self.word_embedding = tf.Variable(tf.random_uniform(shape=[self.vocabulary_size, self.word_embedding_size]))

		self.gru_fwd_input_weights = {
			'r_t' : tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]))]*self.batch_size),
			'z_t' : tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]))]*self.batch_size),
			'h_t' : tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]))]*self.batch_size)
		}
		self.gru_fwd_hidden1_weights = {
			'r_t' : tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]))]*self.batch_size),
			'z_t' : tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]))]*self.batch_size),
			'h_t' : tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]))]*self.batch_size)
		}
		self.gru_fwd_bias = {
			'r_t' : tf.stack([tf.Variable(tf.zeros(shape=[self.word_embedding_size])),]*self.batch_size),
			'z_t' : tf.stack([tf.Variable(tf.zeros(shape=[self.word_embedding_size])),]*self.batch_size),
			'h_t' : tf.stack([tf.Variable(tf.zeros(shape=[self.word_embedding_size])),]*self.batch_size)
		}
		self.gru_bwd_input_weights = {
			'r_t' : tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]))]*self.batch_size),
			'z_t' : tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]))]*self.batch_size),
			'h_t' : tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]))]*self.batch_size)
		}
		self.gru_bwd_hidden1_weights = {
			'r_t' : tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]))]*self.batch_size),
			'z_t' : tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]))]*self.batch_size),
			'h_t' : tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]))]*self.batch_size)
		}
		self.gru_bwd_bias = {
			'r_t' : tf.stack([tf.Variable(tf.zeros(shape=[self.word_embedding_size])),]*self.batch_size),
			'z_t' : tf.stack([tf.Variable(tf.zeros(shape=[self.word_embedding_size])),]*self.batch_size),
			'h_t' : tf.stack([tf.Variable(tf.zeros(shape=[self.word_embedding_size])),]*self.batch_size)
		}
		self.tweet_class = tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.tweet_embedding_size*self.num_classes),shape=[self.tweet_embedding_size,self.num_classes]))]*self.batch_size)
		self.bias_class = tf.stack([tf.zeros(shape=[self.num_classes])]*batch_size)

	def build_model(self):
		train_input = tf.placeholder(tf.int32, shape=[self.batch_size,self.word_max_len])
		train_classes = tf.placeholder(tf.int32, shape=[self.batch_size,self.num_classes])

		word_embedding = tf.embedding_lookup(self.word_embedding,train_input)

		hidden1 = tf.random_normal(shape=[self.batch_size,self.word_embedding_size])
		for t in range(word_max_len):
			rt = tf.nn.sigmoid(tf.nn.l2_normalize(tf.matmul(self.gru_fwd_input_weights['r_t'],word_embedding[:,t]) + tf.matmul(self.gru_fwd_hidden_weights['r_t'],hidden) + self.gru_fwd_bias['r_t'],dim=[0,1]))
			zt = tf.nn.sigmoid(tf.nn.l2_normalize(tf.matmul(self.gru_fwd_input_weights['z_t'],word_embedding[:,t]) + tf.matmul(self.gru_fwd_hidden_weights['z_t'],hidden) + self.gru_fwd_bias['z_t'],dim=[0,1]))
			hid = tf.nn.tanh(tf.nn.l2_normalize(tf.matmul(self.gru_fwd_input_weights['h_t'],word_embedding[:,t]) + tf.matmul(self.gru_fwd_hidden_weights['h_t'],tf.matmul(hidden,rt)) + self.gru_fwd_bias['h_t'],dim=[0,1]))
			hidden = tf.matmul((1 - zt),hidden) + tf.matmul(zt,hid)

		hidden1 = tf.random_normal(shape=[self.batch_size,self.word_embedding_size])
		for t in range(word_max_len):
			rt = tf.nn.sigmoid(tf.nn.l2_normalize(tf.matmul(self.gru_bwd_input_weights['r_t'],word_embedding[:,word_max_len - t - 1]) + tf.matmul(self.gru_bwd_hidden1_weights['r_t'],hidden1) + self.gru_bwd_bias['r_t'],dim=[0,1]))
			zt = tf.nn.sigmoid(tf.nn.l2_normalize(tf.matmul(self.gru_bwd_input_weights['z_t'],word_embedding[:,word_max_len - t - 1]) + tf.matmul(self.gru_bwd_hidden1_weights['z_t'],hidden1) + self.gru_bwd_bias['z_t'],dim=[0,1]))
			hid = tf.nn.tanh(tf.nn.l2_normalize(tf.matmul(self.gru_bwd_input_weights['h_t'],word_embedding[:,word_max_len - t - 1]) + tf.matmul(self.gru_bwd_hidden1_weights['h_t'],tf.matmul(hidden1,rt)) + self.gru_bwd_bias['h_t'],dim=[0,1]))
			hidden1 = tf.matmul((1 - zt),hidden1) + tf.matmul(zt,hid)

		tweet_embedding = tf.matmul(self.grum_weights,hidden) + tf.matmul(self.gru1_weights,hidden1)
		loss = -tf.nn.softmax_cross_entropy_with_logits(labels=train_classes,logits=(tf.matmul(tweet_class,self.tweet_embedding) + self.bias_class)) + 

