import tensorflow as tf
import json
import math
import numpy as np

def read_data(filename):
	data = {}
	with open(filename) as datafile:
		data = json.load(datafile)
	return data

# assume we have tokens at this point in tokenList as a dictionary with counts, tweets as sentences in tweetList

def binary(count,vector):
	i = 0
	while(count > 0 and i <= len(vector)):
		if(count % 2 == 1):
			vector[count] = 1
		count = count / 2
		i = i + 1
	return vector

size = 0

def build_data(tokenList):
	global size
	size = len(tokenList)
	encoding_size = math.ceil(math.log(size,2))
	count = 0
	word2binary = dict()
	for token in tokenList:
		t = np.zeros(shape=(encoding_size),dtype=np.bool)
		word2binary[token] = binary(len(word2binary),encoding_size,t) 
	binary2word = dict(zip(word2binary.values(),word2binary.keys()))
	return word2binary,binary2word,encoding_size

def build_training(word2binary, binary2word, tokenList, tweetList, window_size):
	context = []
	token = []
	for tweet in tweetList:
		extras = list()
		for i in window_size:
			extras.append("UNK")
		ttweet = extras + ttweet + extras
		for j in range(size(tweet)):
			main_token = ttweet[j+window_size]
			context_tokens = list()
			for i in range(1,window_size+1):
				context_tokens.append(ttweet[j+window_size-i])
				context_tokens.append(ttweet[j+window_size+i])
			for context_token in context_tokens:
				token.append(word2binary[main_token])
				context.append(word2binary[context_token])
	return token,context

embedding_size = 256

def generate_batch(batch_size,encoding_size):
	global token, context
	batch = np.ndarray(shape=(batch_size,encoding_size))
	label = np.ndarray(shape=(batch_size,encoding_size,1))
	skip_list = []
	random_input = np.floor(np.random.rand(batch_size)*len(token))
	count = 0
	for i in random_input:
		batch[count] = token[i]
		label[count] = context[i]
		count = count + 1
	return batch,label

learning_rate = 1.0
batch_size = 64

graph = tf.Graph()


with graph.as_default():
	train_inputs = tf.placeholder(tf.bool, shape=(batch_size,encoding_size))
	train_labels = tf.placeholder(tf.bool, shape=(batch_size,encoding_size,1))
	valid_dataset = tf.constant(valid_examples, dtype=tf.bool)

	embeddings = tf.Variable(tf.random_uniform([encoding_size,embedding_size]))
	embed = tf.nn.embedding_lookup(embeddings,train_inputs)

	nce_weights = tf.Variable(tf.truncated_normal([encoding_size,embedding_size],1.0/math.sqrt(embedding_size)))
	nce_biases = tf.Variable(tf.zeros([encoding_size]))

	loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights, biases = nce_biases, labels = train_labels, inputs = embed, num_sampled = num_sampled, num_classes=size))

	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims = True))
	normalized_embeddings = embeddings / norm
	valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
	similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

	init = tf.global_variables_initializer()

num_steps = 100001

with tf.Session(graph=graph) as session:
	init.run()
	print("Initialized")

	average_loss = 0
	for step in range(num_steps):
		batch_inputs, batch_labels = generate_batch(batch_size, encoding_size)
		feed_dict = {train_inputs: batch_inputs, train_labels : batch_labels}
		_,loss_val = session.run([optimizer,loss], feed_dict=feed_dict)
		average_loss += loss_val

		if step%1000 == 0 : 
			sim = similarity.eval()
			top_k = 8
			nearest = (-sim[i,:]).argsort()[1:top_k + 1]
			for k in range(top_k):
				print(binary2word[nearest[k]])