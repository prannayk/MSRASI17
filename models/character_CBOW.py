import tensorflow as tf
import numpy as np
import re
import math
import time

print("Loading NLTK")
from nltk.corpus import brown, reuters, twitter_samples, stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.lancaster import LancasterStemmer
print("Loaded NLTK")
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)
stemmer = LancasterStemmer()
stoplist = stopwords.words('english')

character_window = 2 # >= 1

def batch_normalize(X, eps=1e-8):
	if X.get_shape().ndims == 4:	
		X = tf.nn.l2_normalize(X, [0,1,2], epsilon=eps)
	elif X.get_shape().ndims == 2:	
		X = tf.nn.l2_normalize(X, 0, epsilon=eps)
	elif X.get_shape().ndims == 3:	
		X = tf.nn.l2_normalize(X, [0,1], epsilon=eps)
	elif X.get_shape().ndims == 5:	
		X = tf.nn.l2_normalize(X, [0,1,2,3], epsilon=eps)
	else:
		raise NotImplemented
	return X

flag = True

import string
punctuation = string.punctuation

print("Loading tweets")
f = open("../../training.1600000.processed.noemoticon.csv")
text = f.readlines()
tweetList = list()
for line in text:
	tweetList.append(tknzr.tokenize(line.split(",")[5].decode("utf-8","ignore").encode("utf-8")))
	if len(tweetList) >= 50000 : 
		break
print("Loaded tweets")

maxlen = 0
maxlen_upper_limit = 50
maxsize_upper_limit = 50

print("Loaded from file")
print("Loading Brown corpus")
brownsents = brown.sents()
len_brown_sents = len(brownsents)
brownsentences = []
for bs in brownsents:
	sentence = []
	for token in bs:
		if token in punctuation:
			continue
		sentence.append(token)
	brownsentences.append(sentence)

print("Loading Reuters corpus")
reutersents = reuters.sents()
len_reuters_sents = len(reutersents)
reutersentences = []
for bs in reutersents:
	sentence = []
	for token in bs:
		if token in punctuation:
			continue
		sentence.append(token)
	reutersentences.append(sentence)

print("Loading Twitter corpus")
sample_tweets = []
for tweet in twitter_samples.strings():
	sample_tweets.append(tweet)
tweetList += sample_tweets

print("Loaded everything")


def process_tweets(tweetList, threshold_prob):
	tokenList = dict()
	tokenList['UNK'] = 1
	total = 0
	for tweets in tweetList:
		for token in tweets:
			if token in stoplist:
				continue
			elif token in punctuation:
				continue
			total += 1
			stemmed = stemmer.stem(token)
			if token in tokenList:
				tokenList[token] += 1
			else:
				tokenList[token] = 0
	tokenL = dict(tokenList)
	for token in tokenList: 
		if tokenList[token] < total*threshold_prob :
			tokenList['UNK'] += tokenList[token]
			del tokenL[token]
		elif (token == 'UNK') : 
			continue
	return tokenL

print("Read and processed tweets and tokens")
tokenList = process_tweets(tweetList, 1e-7)
print("Done with tweetList")
browntokens = [i for i in brown.words()]
reutertokens = [ i for i in reuters.words()]
print("Mering: ")

def merge(first_list, second_list):
	return first_list + list(set(second_list) - set(first_list))

tokenList = merge(tokenList.keys(), merge(browntokens, reutertokens))

print("Built dataset of tweets for learning")

def build_data(tokenList):
	global vocabulary_size
	vocabulary_size = len(tokenList)
	word2count = dict()
	binary2word = dict()
	for token in tokenList:
		word2count[token] = len(word2count)
		binary2word[word2count[token]] = token 
	return binary2word,word2count

count2word,word2count = build_data(tokenList)
print(len(tokenList))
vocabulary_size = len(tokenList)
print("Built encodings for tokens")

char2cencoding = dict()
char2cencoding[' '] = len(char2cencoding)
cencoding2char = dict()
cencoding2char[char2cencoding[' ']] = ' '

maxsize = 0
window_size = 5

for tweet in tweetList:
	if len(tweet) > maxlen:
		if len(tweet) > maxlen_upper_limit:
			del tweet
			continue
		else:
			maxlen = len(tweet)
for token in tokenList:
	if len(token) > maxsize:
		if len(token) > maxsize_upper_limit:
			del token
			continue
		else:
			maxsize = len(token)
			print(maxsize)
	new_token = list(token)
	for char in new_token:
		if not char in char2cencoding:
			char2cencoding[char] = len(char2cencoding)
			cencoding2char[char2cencoding[char]] = char

print("Built encoding maps for Characters")
word_max_len = maxlen
char_max_len = maxsize
print("The said word_max_len %d and the said character max_len %d are constants"%(word_max_len, char_max_len))
total_size = len(tweetList)
batch_size = 100
char_size = len(char2cencoding)

def generate_batch(splice):
	global tweetList, batch_size, char2cencoding, word2count
	global char_max_len, word_max_len, flag
	batch = tweetList[splice*batch_size:splice*batch_size +  batch_size]
	train_word = np.ndarray([batch_size,word_max_len],dtype=np.int32)
	train_chars = np.ndarray([batch_size,word_max_len, char_max_len])
	count = 0
	for tweet in batch:
		tokens = tweet
		for t in range(word_max_len):
			if t >= len(tokens):
				train_word[count, t] = word2count['UNK']
				train_chars[count, t] = np.zeros_like(train_chars[count,t])
			else:
				if tokens[t] in word2count:
					train_word[count, t] = word2count[tokens[t]]
				else:
					train_word[count, t] = word2count['UNK']
				for index in range(min(char_max_len, len(tokens[t]))):
					train_chars[count,t,index] = char2cencoding[tokens[t][index]]
				for index in range(len(tokens[t]), char_max_len):
					train_chars[count,t,index] = char2cencoding[' ']
		count += 1
	return train_word, train_chars

def generate_batch_brown(splice):
	global tweetList, batch_size, char2cencoding, word2count
	global char_max_len, word_max_len, flag
	batch = brownsents[splice*batch_size:splice*batch_size +  batch_size]
	train_word = np.ndarray([batch_size,word_max_len],dtype=np.int32)
	train_chars = np.ndarray([batch_size,word_max_len, char_max_len])
	count = 0
	for tweet in batch:
		tokens = tweet
		for t in range(word_max_len):
			if t >= len(tokens):
				train_word[count, t] = word2count['UNK']
				train_chars[count, t] = np.zeros_like(train_chars[count,t])
			else:
				if tokens[t] in word2count:
					train_word[count, t] = word2count[tokens[t]]
				else:
					train_word[count, t] = word2count['UNK']
				for index in range(min(char_max_len, len(tokens[t]))):
					train_chars[count,t,index] = char2cencoding[tokens[t][index]]
				for index in range(len(tokens[t]), char_max_len):
					train_chars[count,t,index] = char2cencoding[' ']
		count += 1
	return train_word, train_chars

def generate_batch_reuters(splice):
	global tweetList, batch_size, char2cencoding, word2count
	global char_max_len, word_max_len, flag
	batch = reutersents[splice*batch_size:splice*batch_size +  batch_size]
	train_word = np.ndarray([batch_size,word_max_len],dtype=np.int32)
	train_chars = np.ndarray([batch_size,word_max_len, char_max_len])
	count = 0
	for tweet in batch:
		tokens = tweet
		for t in range(word_max_len):
			if t >= len(tokens):
				train_word[count, t] = word2count['UNK']
				train_chars[count, t] = np.zeros_like(train_chars[count,t])
			else:
				if tokens[t] in word2count:
					train_word[count, t] = word2count[tokens[t]]
				else:
					train_word[count, t] = word2count['UNK']
				for index in range(min(char_max_len, len(tokens[t]))):
					train_chars[count,t,index] = char2cencoding[tokens[t][index]]
				for index in range(len(tokens[t]), char_max_len):
					train_chars[count,t,index] = char2cencoding[' ']
		count += 1
	return train_word, train_chars

class embeddingCoder():
	def __init__(self,learning_rate, dim1, dim2, dim3,char_embedding_size,word_embedding_size, char_max_len, word_max_len, vocabulary_size, char_size, batch_size,beta, valid_words, valid_chars ):
		self.learning_rate = learning_rate
		self.dim1 = dim1
		self.dim2 = dim2
		self.dim3 = dim3
		self.char_embedding_size = char_embedding_size
		self.word_embedding_size = word_embedding_size
		self.char_max_len = char_max_len
		self.word_max_len = word_max_len
		self.vocabulary_size = vocabulary_size
		self.char_size = char_size
		self.batch_size = batch_size
		self.beta = beta
		self.valid_words = valid_words
		self.valid_chars = valid_chars
		# variables
		with tf.device("/cpu:00"):
			self.char_embeddings = tf.Variable(tf.random_normal(shape=[char_size, char_embedding_size],stddev=1.0))
			self.word_embeddings = tf.Variable(tf.random_normal(shape=[vocabulary_size, word_embedding_size], stddev=1.0))
			# attention matrix
			weight1 = tf.Variable(tf.random_normal(shape=[char_embedding_size,self.dim1]))
			weight2 = tf.Variable(tf.random_normal(shape=[self.dim1,self.dim2]))
			weight3 = tf.Variable(tf.random_normal(shape=[self.dim2,self.dim3]))
			self.weights1 = tf.stack([[weight1]*word_max_len]*batch_size)
			self.weights2 = tf.stack([[weight2]*word_max_len]*batch_size)
			self.weights3 = tf.stack([[weight3]*word_max_len]*batch_size)

	def embedding_creator(self,train_chars, train_words):
		with tf.device("/cpu:0"):
			words = tf.nn.embedding_lookup(self.word_embeddings,train_words)
			chars = tf.nn.embedding_lookup(self.char_embeddings,train_chars)

			character_embedding = tf.reduce_mean(chars, axis=2)
			complete_embedding = tf.nn.l2_normalize(character_embedding + words,1,epsilon=1e-8)
			# known = complete embedding
			contextvector_list = list()
			for i in range(word_max_len):
				count = 0
				contextvector = None
				if i - 1 >= 0:
					contextvector = complete_embedding[:,i - 1]
					count += 1
				if i + 1 < word_max_len:
					if contextvector == None:
						contextvector = complete_embedding[:,i + 1]
					else:
						contextvector += complete_embedding[:,i + 1]
					count += 1

				for j in range(1,character_window):
					if i - j - 1 >= 0:
						contextvector += complete_embedding[:,i -j -1]
						count += 1
					elif i +j + 1 < word_max_len:
						contextvector += complete_embedding[:,i + j + 1]
						count += 1
				contextvector_list.append(contextvector / count)
			context = tf.stack(contextvector_list,axis=1)
			return context, complete_embedding

	def build_model(self):
		with tf.device("/cpu:0"):
			train_chars = tf.placeholder(tf.int32, shape=[self.batch_size, self.word_max_len, self.char_max_len])
			train_words = tf.placeholder(tf.int32, shape=[self.batch_size, self.word_max_len])
			self.train_chars = train_chars
			self.train_words = train_words
			context,complete_embedding = self.embedding_creator(train_chars,train_words)
			# loss = complete_embedding
			r = batch_normalize(tf.matmul(context, complete_embedding, transpose_a=True))
			p = tf.log(tf.nn.softmax(r))
			loss = -tf.reduce_mean(p)

			optimizer = tf.train.AdamOptimizer(self.learning_rate,self.beta).minimize(loss)

			norm = tf.sqrt(tf.reduce_sum(tf.square(self.word_embeddings),1,keep_dims=True))
			normalized_embeddings_word = tf.stack(self.word_embeddings / norm)
			norm = tf.sqrt(tf.reduce_sum(tf.square(self.char_embeddings),1,keep_dims=True))
			normalized_embeddings_char = tf.stack(self.char_embeddings / norm)
			valid_words = tf.placeholder(tf.int32, shape=[self.batch_size, self.word_max_len])
			valid_chars = tf.placeholder(tf.int32, shape=[self.batch_size, self.word_max_len, self.char_max_len])	
			_,valid_embeddings = self.embedding_creator(valid_chars,valid_words)
			valid_run = tf.reshape(valid_embeddings, shape=[self.batch_size, self.word_max_len, 1, self.word_embedding_size])
			# similarity = tf.matmul(valid_run, normalized_embeddings_word, transpose_b=True)
			words_matrix = tf.reshape(tf.transpose(normalized_embeddings_word), shape=[1,1,self.word_embedding_size,self.vocabulary_size])
			similarity = tf.nn.conv2d(valid_run, words_matrix, padding='SAME', strides = [1,1,1,1])
			self.saver = tf.train.Saver()
			self.init = tf.global_variables_initializer()
			self.validwords = valid_words
			self.v_chars = valid_chars
			return optimizer, loss, train_words, train_chars, valid_words, valid_chars, similarity, (self.word_embeddings,self.char_embeddings) , (normalized_embeddings_word, normalized_embeddings_char)

	def initialize(self):
		self.init.run()
	def session(self):
		self.session = tf.InteractiveSession()
		return self.session
	def save(self):
		url = self.saver.save(self.session,'./embedding.ckpt')
		print("Saved in: %s"%(url))
	def restore(self):
		self.saver.restore(self.session, './embedding.ckpt')
		print("Restored model")
	def train(batch):
		self.index += 1
		feed_dict = {
			self.train_words : batch[0],
			self.train_chars : batch[1]
		}
		_,loss_val = self.session.run([optimizer, loss], feed_dict=feed_dict)
		self.average_loss += loss_val
		if self.index % 10 == 0 and self.index > 0:
			print("Average loss is: %s"%(self.average_loss/10))
			self.average_reset()

	def reset():
		self.index = 0
	def average_reset():
		self.average_loss = 0

	def validate(batch):
		feed_dict = {
			self.validwords : batch[0],
			self.v_chars : batch[1]
		}
		word_list = session.run(similarity, feed_dict=feed_dict)
		for t in range(len(word_list)):
			for l in range(min(len(word_list[t]),5)):
				petrol = -word_list[t][l]
				word = petrol[0].argsort()[1]
				print("Said word %s is similar to word %s"%(count2word[valid_words[t,l]],count2word[word]))


num_steps = total_size // batch_size

print("Entering Embedding maker")
embeddingEncoder = embeddingCoder(
		learning_rate = 1e-3,
		dim1 = 64, dim2=16, dim3=1, 
		char_embedding_size = 128,
		word_embedding_size = 128,
		char_max_len = char_max_len,
		word_max_len = word_max_len,
		batch_size = batch_size,
		char_size = char_size,
		vocabulary_size = vocabulary_size,
		beta = 0.001,
		valid_words = valid_words,
		valid_chars = valid_chars
	)
print("Building model")
optimizer, loss, train_words, train_chars, validwords, v_chars, similarity, embedding, norm_embedding = embeddingEncoder.build_model()
print("Setting up session")
session = embeddingEncoder.session()
print("Running init")
embeddingEncoder.initialize()

num_steps_brown = len_brown_sents // batch_size
num_steps_reuters = len_reuters_sents // batch_size

print("Variables Initialized")
print("Running for brown and reuters")
average_loss = 0
count = 0
epoch = 0
valid_brown = generate_batch_brown(np.random.randint(1,20))
start_time = time.time()
embeddingEncoder.reset()
for step in range(num_steps_brown):
	embeddingEncoder.average_reset()
	embeddingEncoder.train(generate_batch_brown(step))
	if step % 10 == 0 and step > 0:
		print("Done with %d tweets:"%(step*batch_size))
		print(time.time() - start_time)
		start_time = time.time()
	if step % 250 == 0 and step > 0:
		embeddingEncoder.validate(valid_brown)
		print("Printing similar words")
embeddingEncoder.save()

print("Running for reuters")
average_loss = 0
count = 0

valid_reuters = generate_batch_reuters(np.random.randint(1,20))
start_time = time.time()
embeddingEncoder.reset()
for step in range(num_steps_reuters):
	embeddingEncoder.average_reset()
	embeddingEncoder.train(generate_batch_reuters(step))
	if step % 10 == 0 and step > 0:
		print("Done with %d tweets:"%(step*batch_size))
		print(time.time() - start_time)
		start_time = time.time()
	if step % 250 == 0 and step > 0:
		embeddingEncoder.validate(valid_reuters)
		print("Printing similar words")
embeddingEncoder.save()

valid_tweets = generate_batch(np.random.randint(1,100))
num_epoch = 3
for epoch in range(num_epoch):
	average_loss = 0
	count = 0
	start_time = time.time()
	embeddingEncoder.reset()
	for step in range(num_steps_):
		embeddingEncoder.average_reset()
		embeddingEncoder.train(generate_batch(step))
		if step % 10 == 0 and step > 0:
			print("Done with %d tweets:"%(step*batch_size))
			print(time.time() - start_time)
			start_time = time.time()
		if step % 250 == 0 and step > 0:
			embeddingEncoder.validate(valid_tweets)
			print("Printing similar words")
	embeddingEncoder.save()
	final_embeddings = embedding.eval()
