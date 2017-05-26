import tensorflow as tf
import numpy as np
import re
import math

character_window = 2 # >= 1

def process_tweet(plain_tweet):
	tokens = plain_tweet.split(" ")
	processed_tokens = list()
	for token in tokens:
		processed_token = re.sub('https?:\/\/.*[\r\n]*','',token)
		processed_token = token.lower()
		processed_tokens.append(processed_token)
	tweet = list()
	for token in processed_tokens:
		tweet.append(token)
	return tweet

f = open("../../Downloads/training.1600000.processed.noemoticon.csv")
text = f.readlines()
tweetList = list()
for line in text:
	tweetList.append(process_tweet(line.split(",")[5]))

maxlen = 0

def process_tweets(tweetList, threshold_prob):
	tokenList = dict()
	tokenList['UNK'] = 1
	for tweets in tweetList:
		for token in tweets:
			if token in tokenList:
				tokenList[token] += 1
			else:
				tokenList[token] = 0
	tokenListCopy = dict(tokenList)
	for token in tokenList:
		if token == 'UNK':
			continue
		if (tokenList[token] / len(tokenList)) < threshold_prob:
			tokenListCopy['UNK'] += tokenList[token]
			del tokenListCopy[token]
	return tokenListCopy

print("Read and processed tweets and tokens")

threshold_prob = 0.001
tokenList = process_tweets(tweetList, threshold_prob)

print("Built dataset of tweets for learning")

vocabulary_size = 0

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
print("Built encodings for tokens")

char2cencoding = dict()
cencoding2char = dict()
maxsize = 0
window_size = 5

for tweet in tweetList:
	for token in tweet:
		if len(token) > maxsize:
			maxsize = len(token)
		new_token = list(token)
		for char in new_token:
			if not char in char2cencoding:
				char2cencoding[char] = len(char2cencoding)
				cencoding2char[char2cencoding[char]] = char

print("Built encoding maps for Characters")
word_max_len = maxlen
char_max_len = maxsize
total_size = len(tweetList)
batch_size = 50

def generate_batch(splice):
	global tweetList, batch_size, char2cencoding, word2count
	global char_max_len, word_max_len
	batch = tweetList[splice*batch_size:splice*batch_size +  batch_size]
	train_word = np.ndarray([batch_size,word_max_len])
	train_chars = np.ndarray([batch_size,word_max_len, char_max_len])
	count = 0
	for tweet in batch:
		tokens = tweet.split()
		for t in range(word_max_len):
			if t >= len(tokens):
				train_word[count, t] = word2count['UNK']
			else:
				if tokens[t] in word2count:
					train_word[count, t] = word2count[tokens[t]]
				else:
					train_word[count, t] = word2count['UNK']
			for index in range(len(tokens[t])):
				train_chars[count,t,index] = char2cencoding[tokens[t][index]]
			for index in range(len(tokens[t], char_max_len)):
				train_chars[count,t,index] = char2cencoding[tokens[t][index]]
		count += 1
	return train_word, train_chars



class embeddingCoder():
	def __init__(self,learning_rate, dim1, dim2, dim3, dim_2,dim_4,char_embedding_size,word_embedding_size):
		self.learning_rate = learning_rate
		self.dim1 = dim1
		self.dim2 = dim2
		self.dim3 = dim3
		self.dim_2 = dim_2
		self.dim_4 = dim_4
		self.char_embedding_size = char_embedding_size
		self.word_embedding_size = word_embedding_size

		# variables
		self.char_embeddings = tf.Variable(tf.random_normal(shape=[char_size, char_embedding_size],stddev=1.0))
		self.word_embeddings = tf.Variable(tf.random_normal(shape=[vocabulary_size, word_embedding_size], stddev=1.0))
		# attention matrix
		weight1 = tf.Variable(tf.random_normal(shape=[char_embedding_size,self.dim1]))
		weight2 = tf.Variable(tf.random_normal(shape=[self.dim1,self.dim2]))
		weight3 = tf.Variable(tf.random_normal(shape=[self.dim2,self.dim3]))
		self.weights1 = tf.stack([[weight1]*word_max_len]*batch_size)
		self.weights2 = tf.stack([[weight2]*word_max_len]*batch_size)
		self.weights2 = tf.stack([[weight3]*word_max_len]*batch_size)

	def embedding_creator(self,train_chars, train_words):
		words = tf.nn.embedding_lookup(self.word_embeddings,train_words)
		chars = tf.nn.embedding_lookup(self.char_embeddings,train_chars)

		attention1 = tf.sigmoid(batch_normalize(tf.matmul(chars,self.weights1,transpose_a=True)))
		attention2 = tf.sigmoid(batch_normalize(tf.matmul(attention1,self.weights2)))
		attention3 = tf.sigmoid(batch_normalize(tf.matmul(attention2,self.weights3)))

		character_embedding = tf.reshape(attention3,shape=[batch_size, word_max_len, char_embedding_size])
		complete_embedding = character_embedding + words
		# known = complete embedding
		contextvector_list = list()
		for i in range(word_max_len):
			count = 0
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
		train_chars = tf.Placeholder(tf.int32, shape=[batch_size, word_max_len, char_max_len])
		train_words = tf.Placeholder(tf.int32, shape=[batch_size, word_max_len])

		context,complete_embedding = self.embedding_creator(train_chars,train_words)

		r = tf.matmul(context, complete_embedding, transpose_a=True)
		p = tf.log(tf.nn.softmax(r))
		loss = tf.reduce_mean(p)

		optimizer = tf.train.AdamOptimizer(learning_rate).maximize(loss)

		norm = tf.sqrt(tf.reduce_sum(tf.square(self.word_embeddings),1,keep_dims=True))
		normalized_embeddings_word = self.word_embeddings / norm

		_,valid_embeddings = self.embedding_creator(valid_dataset)
		similarity = tf.matmul(valid_embeddings, normalized_embeddings_word, transpose_b=True)
		self.saver = tf.train.Saver()
		self.init = tf.global_variables_initializer()

		return optimizer, loss, train_words, train_chars

	def initialize():
		init.run()
	def session():
		self.session = tf.InteractiveSession()
		return self.session
	def save():
		url = self.saver.save(self.session,'./embedding.ckpt')
		print("Saved in: %s"%(url))

num_steps = total_size // batch_size

embeddingEncoder = embeddingCoder(
		learning_rate = 1e-3,
		dim1 = 64, dim2=16, dim3=1, 
		char_embedding_size = 128,
		word_embedding_size = 128
	)

optimizer, loss, train_words, train_chars = embeddingEncoder.build_model()
session = embeddingEncoder.session()
embeddingEncoder.initialize()

init.run()
print("Variables Initialized")

for epoch in range(num_epoch):
	average_loss = 0
	count = 0
	for i in range(num_steps):
		print("Running %d"%(i))
		batch = generate_batch(i)
		feed_dict = {
			train_words : batch[0],
			train_chars : batch[1]
		}
		_, loss_val = session.run([optimizer, loss],feed_dict=feed_dict)
		average_loss += loss_val

		if step % 100 == 0 and step > 0:
			average_loss /= 100
			print("Average_loss %s"%(str(average_loss)))
	embeddingEncoder.save()
	final_embeddings = normalized_embeddings_log
