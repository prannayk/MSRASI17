import tensorflow as tf
import math
import numpy as np
import re
import json

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

maxlen = 0

def read_data(filename):
	data = list()
	with open(filename) as datafile:
		lines = datafile.readlines()
	global maxlen
	for line in lines:
		tweet = json.loads(line)
		data.append(process_tweet(tweet['text']))
		if len(data[len(data) - 1]) > maxlen:
			maxlen = len(data[len(data) - 1])
	for tweet in data:
		if len(tweet) < maxlen:
			for i in range(maxlen - len(tweet)):
				tweet.append('UNK')
	return data

tweetList = read_data("tweets.jsonl")
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

threshold_prob = 0.01
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

binary2word,word2count = build_data(tokenList)
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

## building of data not done

print("Built encoding maps for Characters")

vocabulary_embedding_size = 256
char_embedding_size = 64
char_size = len(char2cencoding)
chword_embedding_size = 64
word_tag_window = 3
intermediate_size = 36
num_tags = 10
learning_rate = 5e-3
feature_size = vocabulary_embedding_size + chword_embedding_size

graph = tf.Graph()

lr = math.floor((window_size-1)/2)
batch_size = 32

print(maxlen)
print(maxsize)

with graph.as_default():
	token_inputs = tf.placeholder(shape=(batch_size,maxlen),dtype=tf.int32)
	char_inputs = tf.placeholder(shape=(batch_size, maxlen,maxsize),dtype=tf.int32)
	token_labels = tf.placeholder(shape=(batch_size,maxlen),dtype=tf.int32)

	wordembeddings = tf.Variable(tf.random_uniform([vocabulary_size,vocabulary_embedding_size]),dtype=tf.float32)
	charembeddings = tf.Variable(tf.random_uniform([char_size,char_embedding_size]),dtype=tf.float32)
	
	word_embed = tf.nn.embedding_lookup(wordembeddings,token_inputs)
	charembeddings = tf.nn.embedding_lookup(charembeddings,char_inputs)
	print(word_embed.shape)
	filterweight = tf.Variable(tf.random_uniform([1,window_size,char_embedding_size,chword_embedding_size]))
	filterbias = tf.Variable(tf.random_uniform([chword_embedding_size]))

	convresult = tf.nn.conv2d(charembeddings,filter=filterweight,strides=[1,1,1,1],padding='VALID',data_format='NHWC')
	convresultnew = tf.nn.relu(convresult + filterbias)

	chwordembeddings = tf.reshape(tf.nn.max_pool(convresultnew,ksize=[1,1,maxsize,1],strides=[1,1,maxsize,1],padding='SAME'),[batch_size,maxlen,chword_embedding_size])

	features = tf.concat([word_embed,chwordembeddings],2)

	tagfilterweight = tf.Variable(tf.random_uniform([1,feature_size,intermediate_size]))
	scorefilterweight = tf.Variable(tf.random_uniform([1,intermediate_size,num_tags]))
	tagbiasweight = tf.Variable(tf.zeros([intermediate_size]))
	scorebiasweight = tf.Variable(tf.zeros([num_tags]))

	scoreinput = tf.tanh(tf.nn.conv1d(features,filters= tagfilterweight,stride=1,padding='SAME') + tagbiasweight)
	score = tf.tanh(tf.nn.conv1d(scoreinput,filters= scorefilterweight,stride=1,padding='SAME') + scorebiasweight)

	loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=score,labels=token_labels))

	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	saver = tf.train.Saver()
	normalized_word_embeddings = wordembeddings / tf.sqrt(tf.reduce_sum(tf.square(wordembeddings),1,keep_dims=True))
	normalized_char_embeddings = charembeddings / tf.sqrt(tf.reduce_sum(tf.square(charembeddings),1,keep_dims=True))
	init = tf.global_variables_initializer()

num_steps = 100000

print("Built Graph, exiting")

with tf.session() as session:
	init.run()
	print("Initialized")

	average_loss = 0
	min_loss = 100000000
	count = 0
	for step in range(num_steps):
		traindata = generate_batch(batch_size)
		feed_dict = {
			token_input : traindata[0],
			char_inputs : traindata[1],
			token_labels : traindata[3]
		}
		_,loss_val = session.run([optimizer,loss], feed_dict=feed_dict)
		average_loss += loss_val

		if step%1000 == 0 and step > 0:
			average_loss /= 1000
			if average_loss < min_loss:
				min_loss = average_loss
				print("Minimum loss till this point: " + str(average_loss))
			print("Loss: " + str(average_loss) + " where step is: " + str(step))
	save_path = saver.save(session, "./convnet.ckpt")
	print("Model saved in file %s" % save_path)
	finalwordembeddings,finalcharembeddings = normalized_word_embeddings.eval(),normalized_char_embeddings.eval()