		

import tensorflow as tf
import numpy as np
import re
import math
import json
import time

print("Loading NLTK")
from nltk.corpus import brown, reuters, twitter_samples, stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.lancaster import LancasterStemmer
import string
import sets
print("Loaded NLTK")
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)
stemmer = LancasterStemmer()
st = LancasterStemmer()
stoplist = stopwords.words('english')
punctuation = string.punctuation
printable = set(string.printable)

query_words = ['need','resources','require','requirement','want']
query_tokens = map(lambda x: st.stem(x).lower(),query_words)

skip_window = 2 # >= 1

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

def token_processor(sentences):
	tokens = []
	return reduce(lambda x, y: merge(x,y) ,sentences)

def sentence_processor(sentence_list):
	return map(lambda y: map(lambda z: re.sub('[%s]*'%(punctuation),'',st.stem(z).lower()) , filter(lambda x:  re.sub(('[%s]*'%(punctuation)),'',x) != '' and (not st.stem(x) in stoplist) and len(st.stem(x)) != 1 , y)), sentence_list)

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
			ptoken = re.sub(('[%s]*'%(punctuation)),' ',token).split(" ")[0]
			stemmed = stemmer.stem(ptoken).lower()
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

def merge(first_list, second_list):
	return first_list + list(set(second_list) - set(first_list))

def filter_fn(x):
	t = re.sub(('[%s]*'%(punctuation)),'',x)
	if t == '':
		return False
	if len(t) == 1:
		return False
	if x in stoplist or t in stoplist:
		return False
	if 'www' in x or 'http' in x:
		return False
	return True

def build_data(tokenList):
	global vocabulary_size
	vocabulary_size = len(tokenList)
	word2count = dict()
	binary2word = dict()
	for token in tokenList:
		word2count[token] = len(word2count)
		binary2word[word2count[token]] = token 
	return binary2word,word2count

flag = True

print("Loading tweets")
f = open("../dataset/nepal.jsonl")
text = f.readlines()
tweetList = list()
reverseListing = dict()
count = 0
for line in text:
	tweet = json.loads(line)
	tweetList.append(tknzr.tokenize(filter(lambda x: x in printable,tweet['text']).decode('utf-8','ignore')))
	reverseListing[count] = tweet['id']
	count += 1
print("Loaded tweets")

maxlen = 0
maxlen_upper_limit = 50
maxsize_upper_limit = 50

print("Loaded from file")
print("Loading Brown corpus")
brownsentences = sentence_processor([i for i in brown.sents()])
len_brown_sents = len(brownsentences)
print("Loading Reuters corpus")
reutersentences = sentence_processor([i for i in reuters.sents()])
len_reuters_sents = len(reutersentences)
print("Loading Twitter corpus")
tweetList = sentence_processor(tweetList)
original_tweets = list(tweetList)
tweetList += sentence_processor([i for i in twitter_samples.strings()])
print("Loaded everything")
print("Read and processed tweets and tokens")
tokenList = process_tweets(tweetList, 1e-7)
print("Done with tweetList")	
browntokens = token_processor(brownsentences)
reutertokens = token_processor(reutersentences)
print("Merging: ")

tokenList = list(set(browntokens + reutertokens + tokenList.keys() + query_tokens) - set(stoplist))
print("Processing tokens")
tokenList = map(lambda x: re.sub('[%s]*'%(punctuation),'',x), filter(lambda x: filter_fn(x) ,tokenList))
brownsentences = map(lambda y: filter(lambda x: filter_fn(x),y), brownsentences)
reutersentences = map(lambda y: filter(lambda x: filter_fn(x),y), reutersentences)
tweetList = map(lambda y: filter(lambda x: filter_fn(x),y), tweetList)
original_tweets = map(lambda y: filter(lambda x: filter_fn(x),y), original_tweets)
print("Built dataset of tweets for learning")
count2word,word2count = build_data(tokenList)
vocabulary_size = len(word2count)
print("Built encodings for tokens")

char2cencoding = dict()
char2cencoding[' '] = len(char2cencoding)
char2cencoding['-'] = len(char2cencoding)
cencoding2char = dict()
cencoding2char[char2cencoding[' ']] = ' '
cencoding2char[char2cencoding['-']] = '-'

maxsize = 0
window_size = 5

for tweet in tweetList:
	if len(tweet) > maxlen:
		if len(tweet) > maxlen_upper_limit:
			del tweet
			continue
		else:
			maxlen = len(tweet)
for tweet in original_tweets:
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

def convert2embedding(batch):
	global tweetList, batch_size, char2cencoding, word2count
	global char_max_len, word_max_len, flag
	train_word = np.ndarray([len(batch),word_max_len],dtype=np.int32)
	train_chars = np.ndarray([len(batch),word_max_len, char_max_len])
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
					if tokens[t][index] in punctuation:
						train_chars[count, t , index] = char2cencoding['-']
					else:
						train_chars[count,t,index] = char2cencoding[tokens[t][index]]
				for index in range(len(tokens[t]), char_max_len):
					train_chars[count,t,index] = char2cencoding[' ']
		count += 1
		if count % 100 == 0 and count > 0:
			print(count)
	return train_word, train_chars

def generate_batch(splice,batch_list):
	batch = batch_list[splice*batch_size:splice*batch_size +  batch_size]
	train_word, train_chars = convert2embedding(batch)
	count = 0
	global word_max_len, word2count
	train_labels = np.ndarray([batch_size, word_max_len, 1])
	for tweet in batch:
		for t in range(word_max_len):
			l = t + np.random.randint(-skip_window, skip_window+1)
			while l >= word_max_len or l < 0:
				l = t + np.random.randint(-skip_window,skip_window+1)
			if l < len(tweet):
				if tweet[l] in word2count:
					train_labels[count,t,0] = word2count[tweet[l]]
				else:
					train_labels[count, t, 0] = word2count['UNK']
			else:
				train_labels[count,t,0] = word2count['UNK']
		count += 1	
	return train_word, train_chars, train_labels


def convert2embedding_classifier(batch):
	return convert2embedding(batch)

def generate_batch_classifier(splice,batch_list,id_list):
	batch = batch_list[splice*batch_size:splice*batch_size +  batch_size]
	train_word,train_chars, train_labels = generate_batch(batch)
	count = 0
	global word_max_len, word2count
	train_labels = np.ndarray([batch_size,3])
	id_list_batch = id_list[splice*batch_size:splice*batch_size + batch_size]
	for tweet in batch:
		if id_list[count] in  avail_tweet:
			train_labels[count] = [0,1,0]
		elif id_list[count] in need_tweet:
			train_labels[count] = [1,0,0]
		else:
			train_labels[count] = [0,0,1]
		count += 1	
	return train_word,train_chars, train_labels, train_labels

class cbow_char():
	def __init__(self,learning_rate, dim1, dim2, dim3,char_embedding_size,word_embedding_size, char_max_len, word_max_len, vocabulary_size, char_size, batch_size,beta, valid_words, valid_chars, num_sampled ):
		self.learning_rate = learning_rate
		self.num_entry = 1
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
		self.num_sampled = num_sampled
		# variables
		with tf.device("/cpu:00"):
			self.char_embeddings = tf.Variable(tf.random_normal(shape=[char_size, char_embedding_size],stddev=1.0/math.sqrt(self.word_embedding_size)))
			self.word_embeddings = tf.Variable(tf.random_normal(shape=[vocabulary_size, word_embedding_size], stddev=1.0/math.sqrt(self.word_embedding_size)))

			self.nce_weight = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.word_embedding_size],stddev = 1.0/math.sqrt(self.word_embedding_size)))
			self.nce_bias = tf.Variable(tf.zeros([self.vocabulary_size]))

			self.grum_weights = tf.Variable(tf.random_normal(stddev=1e2/math.sqrt(self.word_embedding_size*self.tweet_embedding_size),shape=[self.word_embedding_size,self.tweet_embedding_size]),name="gru_weights_1")
			self.gru1_weights = tf.Variable(tf.random_normal(stddev=1e2/math.sqrt(self.word_embedding_size*self.tweet_embedding_size),shape=[self.word_embedding_size,self.tweet_embedding_size]),name="gru_weights_2")


			self.gru_fwd_input_weights = {
				'r_t' : tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]),name="gru_weights_3"),
				'z_t' : tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]),name="gru_weights_4"),
				'h_t' : tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]),name="gru_weights_5")
			}
			self.gru_fwd_hidden_weights = {
				'r_t' : tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]),name="gru_weights_6"),
				'z_t' : tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]),name="gru_weights_7"),
				'h_t' : tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]),name="gru_weights_8")
			}
			self.gru_fwd_bias = {
				'r_t' : tf.Variable(tf.zeros(shape=[self.word_embedding_size])),
				'z_t' : tf.Variable(tf.zeros(shape=[self.word_embedding_size])),
				'h_t' : tf.Variable(tf.zeros(shape=[self.word_embedding_size]))
			}
			self.gru_bwd_input_weights = {
				'r_t' : tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]),name="gru_weights_9"),
				'z_t' : tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]),name="gru_weights_10"),
				'h_t' : tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]),name="gru_weights_11")
			}
			self.gru_bwd_hidden_weights = {
				'r_t' : tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]),name="gru_weights_12"),
				'z_t' : tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]),name="gru_weights_13"),
				'h_t' : tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]),name="gru_weights_14")
			}
			self.gru_bwd_bias = {
				'r_t' : tf.Variable(tf.zeros(shape=[self.word_embedding_size])),
				'z_t' : tf.Variable(tf.zeros(shape=[self.word_embedding_size])),
				'h_t' : tf.Variable(tf.zeros(shape=[self.word_embedding_size]))
			}

			self.tweet_class = tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.tweet_embedding_size*self.num_classes),shape=[self.tweet_embedding_size,self.num_classes]),name="gru_weights_15")
			self.bias_class = tf.zeros(shape=[self.num_classes])

	def embedding_creator(self,train_chars, train_words):
		with tf.device("/cpu:0"):
			words = tf.nn.embedding_lookup(self.word_embeddings,train_words)
			chars = tf.nn.embedding_lookup(self.char_embeddings,train_chars)

			character_embedding = tf.reduce_mean(chars, axis=2)
			complete_embedding = tf.nn.l2_normalize(character_embedding + words,1,epsilon=1e-8)
			
			return complete_embedding

	def gru_embedding(self,word_embedding):
		with tf.device("/gpu:0"):
			tweet_embedding = []
			for batch in range(int(train_input.get_shape()[0])):
				hidden = tf.random_normal(shape=[1,self.word_embedding_size])
				for t in range(self.word_max_len):
					inputv = tf.reshape(word_embedding[batch,t],shape=[1,self.word_embedding_size])
					rt = tf.nn.sigmoid(tf.nn.l2_normalize(tf.matmul(inputv,self.gru_fwd_input_weights['r_t']) + tf.matmul(hidden, self.gru_fwd_hidden_weights['r_t']) + self.gru_fwd_bias['r_t'],dim=[0,1]))
					zt = tf.nn.sigmoid(tf.nn.l2_normalize(tf.matmul(inputv,self.gru_fwd_input_weights['z_t']) + tf.matmul(hidden, self.gru_fwd_hidden_weights['z_t']) + self.gru_fwd_bias['z_t'],dim=[0,1]))
					hid = tf.nn.tanh(tf.nn.l2_normalize(tf.matmul(inputv,self.gru_fwd_input_weights['h_t']) + tf.matmul(hidden*rt,self.gru_fwd_hidden_weights['h_t']) + self.gru_fwd_bias['h_t'],dim=[0,1]))
					hidden = (1 - zt)*hidden + zt*hid
				hidden = tf.random_normal(shape=[1,self.word_embedding_size])
				for t in range(self.word_max_len):
					inputv = tf.reshape(word_embedding[batch,word_max_len - t - 1],shape=[1,self.word_embedding_size])
					rt = tf.nn.sigmoid(tf.nn.l2_normalize(tf.matmul(inputv,self.gru_bwd_input_weights['r_t']) + tf.matmul(hidden,self.gru_bwd_hidden_weights['r_t']) + self.gru_bwd_bias['r_t'],dim=[0,1]))
					zt = tf.nn.sigmoid(tf.nn.l2_normalize(tf.matmul(inputv,self.gru_bwd_input_weights['z_t']) + tf.matmul(hidden,self.gru_bwd_hidden_weights['z_t']) + self.gru_bwd_bias['z_t'],dim=[0,1]))
					hid = tf.nn.tanh(tf.nn.l2_normalize(tf.matmul(inputv,self.gru_bwd_input_weights['h_t']) + tf.matmul(hidden*rt,self.gru_bwd_hidden_weights['h_t']) + self.gru_bwd_bias['h_t'],dim=[0,1]))
					hidden1 = (1 - zt)*hidden + zt*hid
				tweet_embedding.append(tf.transpose(tf.matmul(hidden,self.grum_weights) + tf.matmul(hidden1,self.gru1_weights)))
				print("Rolling..%d"%(batch+1))
			return tf.stack(tweet_embedding)

	def build_model(self):
		with tf.device("/cpu:0"):
			train_chars = tf.placeholder(tf.int32, shape=[self.batch_size, self.word_max_len, self.char_max_len])
			train_words = tf.placeholder(tf.int32, shape=[self.batch_size, self.word_max_len])
			self.train_classes = tf.placeholder(tf.int32, shape=[self.batch_size,self.num_classes])
			train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, self.word_max_len, 1])
			self.train_chars = train_chars
			self.train_words = train_words
			self.train_labels = train_labels
			embedding = self.embedding_creator(train_chars,train_words)

			embedding_trainer = tf.reshape(embedding,shape=[self.batch_size*self.word_max_len,self.word_embedding_size])
			embedding_label = tf.reshape(train_labels, shape=[self.batch_size*self.word_max_len,1])

			p = tf.nn.nce_loss(weights=self.nce_weight,biases=self.nce_bias, labels=embedding_label, inputs=embedding_trainer, num_sampled=self.num_sampled, num_classes=self.vocabulary_size)
			loss = tf.reduce_mean(p)

			optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta).minimize(loss)

			norm = tf.sqrt(tf.reduce_sum(tf.square(self.word_embeddings),1,keep_dims=True))
			normalized_embeddings_word = tf.stack(self.word_embeddings / norm)
			norm = tf.sqrt(tf.reduce_sum(tf.square(self.char_embeddings),1,keep_dims=True))
			normalized_embeddings_char = tf.stack(self.char_embeddings / norm)
			valid_words = tf.placeholder(tf.int32, shape=[self.batch_size, self.word_max_len])
			valid_chars = tf.placeholder(tf.int32, shape=[self.batch_size, self.word_max_len, self.char_max_len])	
			valid_embeddings = self.embedding_creator(valid_chars,valid_words)
			valid_run = tf.reshape(valid_embeddings, shape=[self.batch_size, self.word_max_len, 1, self.word_embedding_size])
			words_matrix = tf.reshape(tf.transpose(normalized_embeddings_word), shape=[1,1,self.word_embedding_size,self.vocabulary_size])
			similarity = tf.nn.conv2d(valid_run, words_matrix, padding='SAME', strides = [1,1,1,1])

			self.saver = tf.train.Saver()
			self.init = tf.global_variables_initializer()
			self.validwords = valid_words
			self.v_chars = valid_chars
			self.optimizer = optimizer
			self.loss = loss
			self.similarity = similarity

			self.tweet_embedding = self.gru_embedding(complete_embedding)
			regularization = tf.nn.l2_loss(gru_weights[0])
			for i in range(1,len(gru_weights)):
				regularization += tf.nn.l2_loss(gru_weights[i])
			self.loss_classifier = -tf.nn.softmax_cross_entropy_with_logits(labels=self.train_classes,logits=tf.reshape(tf.matmul(tf.stack([self.tweet_class]*self.batch_size),self.tweet_embedding,transpose_a=True),shape=[self.batch_size,3]) + tf.stack([self.bias_class]*self.batch_size)) + (0.3*regularization)
			self.optimizer_classifier = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_classifier)

			self.batch_prob_row = tf.nn.softmax(tf.reshape(tf.matmul(tf.stack([self.tweet_class]*self.batch_size),self.tweet_embedding_creator(self.complete_embedding),transpose_a=True),shape=[self.batch_size,self.num_classes]) + tf.stack([self.bias_class]*self.batch_size))
			self.batch_prob = self.batch_prob_row[:,0]

			return optimizer, loss, train_words, train_chars, train_labels, valid_words, valid_chars, similarity, (self.word_embeddings,self.char_embeddings) , (normalized_embeddings_word, normalized_embeddings_char)

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
	def train(self,batch):
		self.index += 1
		feed_dict = {
			self.train_words : batch[0],
			self.train_chars : batch[1],
			self.train_labels : batch[2]
		}
		_,loss_val = self.session.run([self.optimizer, self.loss], feed_dict=feed_dict)
		self.average_loss += loss_val
		if self.index % 10 == 0 and self.index > 0:
			print("Average loss is: %s"%(self.average_loss/10))
			self.average_reset()

	def reset(self):
		self.index = 0
	def average_reset(self):
		self.average_loss = 0

	def validate(self,batch):
		feed_dict = {
			self.validwords : batch[0],
			self.v_chars : batch[1]
		}
		file_text = []
		word_list = session.run(self.similarity, feed_dict=feed_dict)
		for t in range(len(word_list)):
			for l in range(min(len(word_list[t]),5)):
				petrol = -word_list[t][l]
				word = petrol[0].argsort()[1]
				file_text.append("Said word %s is similar to word %s"%(count2word[batch[0][t,l]],count2word[word]))
		filedata = '\n'.join(file_text)
		self.num_entry += 1
		with open("./logz/last_run_%d.txt"%(self.num_entry),mode="w") as fil:
			fil.write(filedata.encode('utf-8','ignore'))

	def train_on_batch(self,num_epoch, batch_list):
		num_step = len(batch_list) // self.batch_size
		validate = generate_batch(np.random.randint(1,20),batch_list)[:2]
		for epoch in range(num_epoch):
			self.reset()
			start_time = time.time()
			for step in range(num_step):
				self.average_reset()
				self.train(generate_batch(step, batch_list))
				if step % 10 == 0 and step > 0:
					print("Done with %d tweets for epoch %d"%(step*self.batch_size,epoch))
					print(time.time()-start_time)
					start_time = time.time()
				if step % 100 == 0 and step > 0:
					self.validate(validate)
			self.save()

	# def create_query(self,num_queries,query_tokens,num_total):
	# 	print("Petrol")
	# 	self.num_queries = num_queries
	# 	self.query_tokens = query_tokens
	# 	query_size = len(query_tokens)
	# 	self.total_batch_size = num_total
	# 	query_tweet_list = []
	# 	for r in range(num_queries):
	# 		l,t = np.random.randint(query_size,size=[2])
	# 		while l == t:
	# 			l,t = np.random.randint(query_size,size=[2])
	# 		query_tweet_list.append([self.query_tokens[l],self.query_tokens[t]])
	# 	self.query_list = convert2embedding(query_tweet_list)
	# 	self.query_lit = list()

	def rank_on_batch(self, batch_list,case):
		print("Getting results")
		ident = str(case) + str(np.random.randint(100))
		batch = convert2embedding(batch_list)
		feed_dict = {
			self.ir_words : batch[0],
			self.ir_chars : batch[1],
			self.query_lit[0] : self.query_list[1],
			self.query_lit[1] : self.query_list[0]
		}
		query_similarity = self.session.run(self.query_similarity,feed_dict=feed_dict)
		sorted_queries = [i for i in sorted(enumerate(query_similarity),key=lambda x: -x[1])]
		text_lines = []
		count = 0
		for t in sorted_queries:
			text_lines.append('%s Q0 %s %d %f %s'%(case,reverseListing[t[0]],count,t[1],ident))
			count += 1
		with open('./skipgram.result.text',mode="w") as f:
			f.write('\n'.join(text_lines))

	def train_classifier(self, batch):
		self.index += 1
		feed_dict = {
			self.train_words : batch[0],
			self.train_chars : batch[1],
			self.train_classes : batch[3],
			self.train_labels :  batch[2]
		}
		_, loss_val = self.session.run([self.optimizer_classifier,self.loss_classifier],feed_dict=feed_dict)
		_ = self.session.run([self.optimizer, self.loss],feed_dict=feed_dict)
		self.average_loss += loss_val
		if self.index % 10 and self.index > 0:
			print("Average loss is: %s"%(self.average_loss/10))
			self.average_reset()

	def validate_classifier(self,batch,id_list):
		feed_dict = {
			self.train_words : batch[0],
			self.train_chars : batch[1],
			self.train_classes : batch[3],
			self.train_labels :  batch[2]
		}
		file_text = []
		prob_row = self.session.run(self.batch_prob_row,feed_dict=feed_dict)
		for i in range(prob_row.shape[0]):
			file_text.append('%s %d %d %d'%(id_list[i],prob_row[i,0],prob_row[i,1],prob_row[i,2]))
		filedata = '\n'.join(file_text)
		self.num_entry += 1	
		with open("./tweet2vec_log/last_run_%d.txt"%(self.num_entry),mode="w") as fil:
			fil.write(filedata.encode('utf-8','ignore'))	

	def train_on_batch_classifier(self,num_epoch, batch_list):
		num_step = len(batch_list) // self.batch_size
		validate = generate_batch_classifier(np.random.randint(1,20),batch_list)[:1]
		validate_id = generate_batch_classifier(np.random.randint(1,20),batch_list)[1]
		for epoch in range(num_epoch):
			self.reset()
			start_time = time.time()
			for step in range(num_step):
				self.average_reset()
				self.train(generate_batch_classifier(step, batch_list)[:1])
				if step % 10 == 0 and step > 0:
					print("Done with %d tweets for epoch %d"%(step*self.batch_size,epoch))
					print(time.time()-start_time)
					start_time = time.time()
				if step % 100 == 0 and step > 0:
					self.validate_classifier(validate,validate_id)
			self.save()

	def rank_on_batch_classifier(self, batch_list,case):
		print("Getting results")
		query_similarity = []
		ident = case + str(np.random.randint(100))
		for l in range(math.ceil(len(batch_list) / 100)):
			batch = convert2embedding_classifier(batch_list[l*100:l*100 + 100])
			feed_dict = {
				self.train_words : batch[0],
				self.train_chars : batch[1]
			}
			query_similarity += self.session(self.batch_prob,feed_dict=feed_dict)
		sorted_queries = [i for i in sorted(enumerate(query_similarity),lambda x: x[1])]
		text_lines = []
		count = 0
		for t in sorted_queries:
			text_lines.append('%s Q0 %s %d %f %s'%(case,reverseTweetList[t[0]],count,t[1],ident))
			count += 1
		with open("./tweet2vec.result.txt",mode="w") as f:
			f.write('\n'.join(text_lines))

print("Entering Embedding maker")
embeddingEncoder = cbow_char(
		learning_rate = 5e-1,
		dim1 = 64, dim2=16, dim3=1, 
		char_embedding_size = 128,
		word_embedding_size = 128,
		char_max_len = char_max_len,
		word_max_len = word_max_len,
		batch_size = batch_size,
		char_size = char_size,
		vocabulary_size = vocabulary_size,
		beta = 0.9,
		valid_words = None,
		valid_chars = None,
		num_sampled = 50
	)

print("Building model")
_ = embeddingEncoder.build_model()
print("Setting up session")
session = embeddingEncoder.session()
print("Running init")
embeddingEncoder.initialize()
print("Variables Initialized")
print("Running for brown and reuters")
print("Running for Brown")
embeddingEncoder.train_on_batch(5,brownsentences)
print("Running for reuters")
embeddingEncoder.train_on_batch(5, reutersentences)
print("Running for tweets")
embeddingEncoder.train_on_batch(5, tweetList)
embeddingEncoder.rank_on_batch(original_tweets, np.random.randint(1e6))
embeddingEncoder.train_on_batch_classifier(5, original_tweets)
embeddingEncoder.rank_on_batch_classifier(original_tweets, np.random.randint(1e6))
