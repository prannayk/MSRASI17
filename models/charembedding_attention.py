import tensorflow as tf
import numpy as np
import re
import math
import time

character_window = 2 # >= 1

def batch_normalize(X, eps=1e-6):
	if X.get_shape().ndims == 4 :
		mean = tf.reduce_mean(X,[0,1,2])
		stddev = tf.reduce_mean(tf.square(X-mean),[0,1,2])
		X = (X - mean)/tf.sqrt(stddev + eps)
	elif X.get_shape().ndims == 2:
		mean = tf.reduce_mean(X,[0])
		stddev = tf.reduce_mean(tf.square(X-mean),[0])
		X = (X - mean)/tf.sqrt(stddev + eps)
	elif X.get_shape().ndims == 5:
		mean = tf.reduce_mean(X,[0,1,2,3])
		stddev = tf.reduce_mean(tf.square(X-mean),[0,1,2,3])
		X = (X-mean)/tf.sqrt(stddev + eps)
	elif X.get_shape().ndims == 3:
		mean = tf.reduce_mean(X,[0,1])
		stddev = tf.reduce_mean(tf.square(X-mean),[0,1])
		X = (X-mean)/tf.sqrt(stddev + eps)
	else:
		raise NoImplementationForSuchDimensions
	return X
flag = True
def process_tweet(plain_tweet):
	global flag
	tokens = plain_tweet.split(" ")
	processed_tokens = list()
	for token in tokens:
		processed_token = re.sub('https?:\/\/.*[\r\n]*','',token)
		processed_token = re.sub('[^a-zA-Z0-9#]','',processed_token)
		processed_token = re.sub('["!]','',processed_token)
		processed_token = processed_token.lower()
		processed_tokens.append(processed_token)
	tweet = list()
	for ptoken in processed_tokens:
		if not ptoken == '':
			tweet.append(ptoken)
	return tweet

f = open("../../training.1600000.processed.noemoticon.csv")
text = f.readlines()
tweetList = list()
for line in text:
	tweetList.append(process_tweet(line.split(",")[5]))

maxlen = 0
maxlen_upper_limit = 50
maxsize_upper_limit = 50

print("Loaded from file")

def process_tweets(tweetList, threshold_prob):
	tokenList = dict()
	tokenList['UNK'] = 1
	total = 0
	for tweets in tweetList:
		for token in tweets:
			total += 1
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
	for token in tweet:
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
batch_size = 20
char_size = len(char2cencoding)

def generate_batch(splice):
	global tweetList, batch_size, char2cencoding, word2count
	global char_max_len, word_max_len, flag
	batch = tweetList[splice*batch_size:splice*batch_size +  batch_size]
	train_word = np.ndarray([batch_size,word_max_len],dtype=np.int32)
	train_chars = np.ndarray([batch_size,word_max_len, char_max_len])
	count = 0
	for tweet in batch:
		if flag:
			print(tweet)
		tokens = tweet
		for t in range(word_max_len):
			if t >= len(tokens):
				
				train_word[count, t] = word2count['UNK']
				train_chars[count, t] = np.zeros_like(train_chars[count,t])
			else:
				if flag:
					print(tokens[t])
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
print("Generating batches")
flag = True
valid_words, valid_chars = generate_batch(np.random.randint(1,100))
flag = False
print(valid_words)
print(valid_chars)

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

			attention1 = tf.sigmoid(batch_normalize(tf.matmul(chars,self.weights1)))
			attention2 = tf.sigmoid(batch_normalize(tf.matmul(attention1,self.weights2)))
			attention3 = tf.sigmoid(batch_normalize(tf.matmul(attention2,self.weights3)))
			hidden_layer = tf.matmul(attention3, chars, transpose_a = True)
			character_embedding = tf.reshape(hidden_layer,shape=[self.batch_size, self.word_max_len, self.char_embedding_size])
			complete_embedding = character_embedding + words
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

			context,complete_embedding = self.embedding_creator(train_chars,train_words)
			# loss = complete_embedding
			r = batch_normalize(tf.matmul(context, complete_embedding, transpose_a=True))
			p = tf.log(tf.nn.softmax(r))
			loss = -tf.reduce_mean(p)

			optimizer = tf.train.AdamOptimizer(self.learning_rate,self.beta).minimize(loss)

			norm = tf.sqrt(tf.reduce_sum(tf.square(self.word_embeddings),1,keep_dims=True))
			normalized_embeddings_word = tf.stack(self.word_embeddings / norm)
			valid_words = tf.placeholder(tf.int32, shape=[self.batch_size, self.word_max_len])
			valid_chars = tf.placeholder(tf.int32, shape=[self.batch_size, self.word_max_len, self.char_max_len])	
			_,valid_embeddings = self.embedding_creator(valid_chars,valid_words)
			valid_run = tf.reshape(valid_embeddings, shape=[self.batch_size, self.word_max_len, 1, self.word_embedding_size])
			# similarity = tf.matmul(valid_run, normalized_embeddings_word, transpose_b=True)
			words_matrix = tf.reshape(tf.transpose(normalized_embeddings_word), shape=[1,1,self.word_embedding_size,self.vocabulary_size])
			similarity = tf.nn.conv2d(valid_run, words_matrix, padding='SAME', strides = [1,1,1,1])
			self.saver = tf.train.Saver()
			self.init = tf.global_variables_initializer()

			return optimizer, loss, train_words, train_chars, valid_words, valid_chars, similarity, (self.word_embeddings,self.char_embeddings)

	def initialize(self):
		self.init.run()
	def session(self):
		self.session = tf.InteractiveSession()
		return self.session
	def save(self):
		url = self.saver.save(self.session,'./embedding.ckpt')
		print("Saved in: %s"%(url))

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
optimizer, loss, train_words, train_chars, validwords, v_chars, similarity, embedding = embeddingEncoder.build_model()
print("Setting up session")
session = embeddingEncoder.session()
print("Running init")
embeddingEncoder.initialize()

print("Variables Initialized")
num_epoch = 10
for epoch in range(num_epoch):
	average_loss = 0
	count = 0
	start_time = time.time()
	for step in range(num_steps):
		batch = generate_batch(step)
		feed_dict = {
			train_words : batch[0],
			train_chars : batch[1]
		}
		_, loss_val = session.run([optimizer, loss],feed_dict=feed_dict)
		average_loss += loss_val

		if step % 10 == 0 and step > 0:
			average_loss /= 10
			print("Done with %d tweets:"%(step*batch_size))
			print("Average_loss %s where the epoch is: %d"%(str(average_loss), epoch))
			print(time.time() - start_time)
			start_time = time.time()
			average_loss = 0
		if step % 100 == 0 and step > 0:
			print("Printing similar words")
			feed_dict = {
				validwords : valid_words,
				v_chars : valid_chars
			}
			word_list = session.run(similarity, feed_dict=feed_dict)
			for t in range(len(word_list)):
				for l in range(min(len(word_list[t]),5)):
					petrol = -word_list[t][l]
					word = petrol[0].argsort()[1]
					print("Said word %s is similar to word %s"%(count2word[valid_words[t,l]],count2word[word]))

	embeddingEncoder.save()
	final_embeddings = embedding.eval()
