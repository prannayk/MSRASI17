import tensorflow as tf
import json
import math
import numpy as np
import re

def process_tweet(plain_tweet):
	tokens = plain_tweet.split(" ")
	processed_tokens = list()
	for token in tokens:
		processed_token = re.sub('https?:\/\/.*[\r\n]*','',token)
		processed_token = re.sub('#','',processed_token)
		processed_token = re.sub('@','',processed_token)
		reg = re.compile('[^a-zA-Z0-9]')
		processed_token = reg.sub('',processed_token)
		processed_token = token.lower()
		processed_tokens.append(processed_token)
	tweet = list()
	for token in processed_tokens:
		tweet.append(token)
	return tweet

def read_data(filename):
	data = list()
	with open(filename) as datafile:
		lines = datafile.readlines()
	for line in lines:
		tweet = json.loads(line)
		data.append(process_tweet(tweet['text']))
	return data


tweetList = read_data("fire2016-final-Nepal-earthquake-tweets.jsonl")
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

def build_training(binary2word, tokenList, tweetList, window_size, word2count):
	context = []
	token = []
	for tweet in tweetList:
		extras = list()
		for i in range(window_size):
			extras.append("UNK")
		ttweet = list(tweet)
		ttweet = extras + ttweet + extras
		for j in range(len(tweet)):
			main_token = ttweet[j+window_size]
			if not main_token in word2count:
				continue
			context_tokens = list()
			for i in range(1,window_size+1):
				context_tokens.append(ttweet[j+window_size-i])
				context_tokens.append(ttweet[j+window_size+i])
			for context_token in context_tokens:
				token.append(word2count[main_token])
				if not context_token in word2count:
					context.append(word2count['UNK'])
				else:
					context.append(word2count[context_token])
	return token,context

window_size = 2
tokens,context_words = build_training(binary2word,tokenList,tweetList, window_size,word2count)
print("Built training dataset")

embedding_size = 256

def generate_batch(batch_size):
	global tokens, context_words
	batch = np.ndarray(shape=(batch_size),dtype=np.int64)
	label = np.ndarray(shape=(batch_size,1),dtype=np.int64)
	skip_list = []
	random_input = np.floor(np.random.rand(batch_size)*len(tokens)).astype(int)
	count = 0
	for i in random_input:
		batch[count] = tokens[i]
		label[count] = context_words[i]
		count = count + 1
	return batch,label

valid_examples,_ = generate_batch(64)
learning_rate = 1.0
batch_size = 64
num_sampled = 32

graph = tf.Graph()


with graph.as_default():
	train_inputs = tf.placeholder(tf.int32, shape=(batch_size))
	train_labels = tf.placeholder(tf.int64, shape=(batch_size,1))
	valid_dataset = tf.constant(valid_examples)

	embeddings = tf.Variable(tf.random_uniform([vocabulary_size,embedding_size]))
	embed = tf.nn.embedding_lookup(embeddings,train_inputs)

	nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size,embedding_size],1.0/math.sqrt(embedding_size)))
	nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

	loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights, biases = nce_biases, labels = train_labels, inputs = embed, num_sampled = num_sampled, num_classes=vocabulary_size))

	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims = True))
	normalized_embeddings = embeddings / norm
	valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
	similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
	saver = tf.train.Saver()
	init = tf.global_variables_initializer()

num_steps = 1000000001

with tf.Session(graph=graph) as session:
	init.run()
	print("Initialized")

	average_loss = 0
	min_loss = 10000000000
	count = 0
	for step in range(num_steps):
		batch_inputs, batch_labels = generate_batch(batch_size)
		feed_dict = {train_inputs: batch_inputs, train_labels : batch_labels}
		_,loss_val = session.run([optimizer,loss], feed_dict=feed_dict)
		average_loss += loss_val
		# print("Running" + str(step))
		if step%1000 == 0 and step >= 2000:
			average_loss /= 1000
			if average_loss < min_loss:
				min_loss = average_loss
				count = 0
				normalized_embeddings_log = normalized_embeddings.eval()
			else:
				count += 1
			if count > 10:
				break
			print("Average loss: " + str(average_loss) + " and Minimum Loss: " + str(min_loss) + "and running count: " + str(count) + "/1000")
			average_loss = 0
	save_path = saver.save(session, "./model.ckpt")
	print("Model saved in file %s" % save_path)
	final_embeddings = normalized_embeddings_log