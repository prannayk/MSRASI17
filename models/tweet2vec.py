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
original_tweets = tweetList
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
	train_word = np.ndarray([batch_size,word_max_len],dtype=np.int32)
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
		count += 1
	return train_word

def generate_batch(splice,batch_list,id_list):
	batch = batch_list[splice*batch_size:splice*batch_size +  batch_size]
	id_list_batch = id_list[splice*batch_size:splice*batch_size + batch_size]
	train_word = convert2embedding(batch)
	count = 0
	global word_max_len, word2count
	train_labels = np.ndarray([batch_size,3])
	for tweet in batch:
		if id_list[count] in  avail_tweet:
			train_labels[count] = [0,1,0]
		else if id_list[count] in need_tweet:
			train_labels[count] = [1,0,0]
		else:
			train_labels[count] = [0,0,1]
		count += 1	
	return train_word, train_chars, train_labels

class tweet2vec():
	def __init__(batch_size, vocabulary_size, word_max_len,word_embedding_size=128, tweet_embedding_size=256, num_classes=3, learning_rate=5e-1,name='tweet2vec.py'):
		self.batch_size = batch_size
		self.word_embedding_size = word_embedding_size
		self.num_classes = num_classes
		self.tweet_embedding_size = tweet_embedding_size
		self.vocabulary_size = vocabulary_size
		self.word_max_len = word_max_len
		self.learning_rate = learning_rate
		self.name = name
		self.num_entry = 0

		self.grum_weights = tf.stack([tf.Variable(tf.random_normal(stddev=1e2/math.sqrt(self.word_embedding_size*self.tweet_embedding_size),shape=[self.word_embedding_size,self.tweet_embedding_size]),name="gru_weights_1")]*self.batch_size)
		self.gru1_weights = tf.stack([tf.Variable(tf.random_normal(stddev=1e2/math.sqrt(self.word_embedding_size*self.tweet_embedding_size),shape=[self.word_embedding_size,self.tweet_embedding_size]),name="gru_weights_2")]*self.batch_size)

		self.word_embedding = tf.Variable(tf.random_uniform(shape=[self.vocabulary_size, self.word_embedding_size]))

		self.gru_fwd_input_weights = {
			'r_t' : tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]),name="gru_weights_3")]*self.batch_size),
			'z_t' : tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]),name="gru_weights_4")]*self.batch_size),
			'h_t' : tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]),name="gru_weights_5")]*self.batch_size)
		}
		self.gru_fwd_hidden1_weights = {
			'r_t' : tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]),name="gru_weights_6")]*self.batch_size),
			'z_t' : tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]),name="gru_weights_7")]*self.batch_size),
			'h_t' : tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]),name="gru_weights_8")]*self.batch_size)
		}
		self.gru_fwd_bias = {
			'r_t' : tf.stack([tf.Variable(tf.zeros(shape=[self.word_embedding_size])),]*self.batch_size),
			'z_t' : tf.stack([tf.Variable(tf.zeros(shape=[self.word_embedding_size])),]*self.batch_size),
			'h_t' : tf.stack([tf.Variable(tf.zeros(shape=[self.word_embedding_size])),]*self.batch_size)
		}
		self.gru_bwd_input_weights = {
			'r_t' : tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]),name="gru_weights_9")]*self.batch_size),
			'z_t' : tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]),name="gru_weights_10")]*self.batch_size),
			'h_t' : tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]),name="gru_weights_11")]*self.batch_size)
		}
		self.gru_bwd_hidden1_weights = {
			'r_t' : tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]),name="gru_weights_12")]*self.batch_size),
			'z_t' : tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]),name="gru_weights_13")]*self.batch_size),
			'h_t' : tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.word_embedding_size),shape=[self.word_embedding_size,self.word_embedding_size]),name="gru_weights_14")]*self.batch_size)
		}
		self.gru_bwd_bias = {
			'r_t' : tf.stack([tf.Variable(tf.zeros(shape=[self.word_embedding_size])),]*self.batch_size),
			'z_t' : tf.stack([tf.Variable(tf.zeros(shape=[self.word_embedding_size])),]*self.batch_size),
			'h_t' : tf.stack([tf.Variable(tf.zeros(shape=[self.word_embedding_size])),]*self.batch_size)
		}

		self.tweet_class = tf.stack([tf.Variable(tf.random_normal(stddev=1.0/math.sqrt(self.tweet_embedding_size*self.num_classes),shape=[self.tweet_embedding_size,self.num_classes]),name="gru_weights_15")]*self.batch_size)
		self.bias_class = tf.stack([tf.zeros(shape=[self.num_classes])]*batch_size)

	def tweet_embedding_creator(self,train_input):
		word_embedding = tf.embedding_lookup(self.word_embedding,train_input)
		tweet_embedding = []
		for batch in range(train_input.shape[0]):
			hidden1 = tf.random_normal(shape=[self.batch_size,self.word_embedding_size])
			for t in range(word_max_len):
				rt = tf.nn.sigmoid(tf.nn.l2_normalize(tf.matmul(self.gru_fwd_input_weights['r_t'],word_embedding[batch,t]) + tf.matmul(self.gru_fwd_hidden_weights['r_t'],hidden) + self.gru_fwd_bias['r_t'],dim=[0,1]))
				zt = tf.nn.sigmoid(tf.nn.l2_normalize(tf.matmul(self.gru_fwd_input_weights['z_t'],word_embedding[batch,t]) + tf.matmul(self.gru_fwd_hidden_weights['z_t'],hidden) + self.gru_fwd_bias['z_t'],dim=[0,1]))
				hid = tf.nn.tanh(tf.nn.l2_normalize(tf.matmul(self.gru_fwd_input_weights['h_t'],word_embedding[batch,t]) + tf.matmul(self.gru_fwd_hidden_weights['h_t'],tf.matmul(hidden,rt)) + self.gru_fwd_bias['h_t'],dim=[0,1]))
				hidden = tf.matmul((1 - zt),hidden) + tf.matmul(zt,hid)

			hidden1 = tf.random_normal(shape=[self.batch_size,self.word_embedding_size])
			for t in range(word_max_len):
				rt = tf.nn.sigmoid(tf.nn.l2_normalize(tf.matmul(self.gru_bwd_input_weights['r_t'],word_embedding[batch,word_max_len - t - 1]) + tf.matmul(self.gru_bwd_hidden1_weights['r_t'],hidden1) + self.gru_bwd_bias['r_t'],dim=[0,1]))
				zt = tf.nn.sigmoid(tf.nn.l2_normalize(tf.matmul(self.gru_bwd_input_weights['z_t'],word_embedding[batch,word_max_len - t - 1]) + tf.matmul(self.gru_bwd_hidden1_weights['z_t'],hidden1) + self.gru_bwd_bias['z_t'],dim=[0,1]))
				hid = tf.nn.tanh(tf.nn.l2_normalize(tf.matmul(self.gru_bwd_input_weights['h_t'],word_embedding[batch,word_max_len - t - 1]) + tf.matmul(self.gru_bwd_hidden1_weights['h_t'],tf.matmul(hidden1,rt)) + self.gru_bwd_bias['h_t'],dim=[0,1]))
				hidden1 = tf.matmul((1 - zt),hidden1) + tf.matmul(zt,hid)

			tweet_embedding.append(tf.matmul(self.grum_weights,hidden) + tf.matmul(self.gru1_weights,hidden1))
		return tf.stack(tweet_embedding)

	def build_model(self):
		self.train_input = tf.placeholder(tf.int32, shape=[self.batch_size,self.word_max_len])
		self.train_classes = tf.placeholder(tf.int32, shape=[self.batch_size,self.num_classes])
		tweet_embedding = self.tweet_embedding_creator(self.train_input)
		# regularization
		regularization = reduce(lambda x,y: tf.nn.l2_loss(y)+x ,[i for i in filter(lambda x: x.name.startswith("gru"),tf.trainable_variables())])
		self.loss = -tf.nn.softmax_cross_entropy_with_logits(labels=self.train_classes,logits=(tf.matmul(self.tweet_class,self.tweet_embedding) + self.bias_class)) + (0.3*regularization)
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

		self.batch_input = tf.placeholder(tf.int32, shape=[self.total_batch_size, self.word_max_len])
		self.batch_prob_row = tf.nn.softmax(self.tweet_embedding_creator(self.batch_input))
		self.batch_prob = self.batch_prob_row[:,0]

		self.saver = tf.train.Saver()
		self.init = tf.global_variables_initializer()

	def initialize(self):
		self.session = tf.train.InteractiveSession()
		self.init.run()
		return session
	def save(self):
		self.saver(self.session,self.name)
		print("Saved at url: %s"%(self.name))
	def restore(self):
		self.saver.restore(self.session,self.name)
		print("Restored model")

	def train(self, batch):
		self.index += 1
		feed_dict = {
			self.train_input : batch[0],
			self.train_classes : batch[1]
		}
		_, loss_val = self.session.run([self.optimizer,self.loss],feed_dict=feed_dict)
		self.average_loss += loss_val
		if self.index % 10 and self.index > 0:
			print("Average loss is: %s"%(self.average_loss/10))
			self.average_reset()

	def reset(self):
		self.index = 0
	def average_reset(self):
		self.average_reset = 0

	def validate(self,batch,id_list):
		feed_dict = {
			self.batch_input : batch[0]
		}
		file_text = []
		prob_row = self.session.run(self.batch_prob_row,feed_dict=feed_dict)
		for i in range(prob_row.shape[0]):
			file_text.append('%s %d %d %d'%(id_list[i],prob_row[i,0],prob_row[i,1],prob_row[i,2]))
		filedata = '\n'.join(file_text)
		self.num_entry += 1	
		with open("./tweet2vec_log/last_run_%d.txt"%(self.num_entry),mode="w") as fil:
			fil.write(filedata.encode('utf-8','ignore'))	

	def train_on_batch(self,num_epoch, batch_list):
		num_step = len(batch_list) // self.batch_size
		validate = generate_batch(np.random.randint(1,20),batch_list)[:2]
		validate_id = generate_batch(np.random.randint(1,20),batch_list)[2]
		for epoch in range(num_epoch):
			self.reset()
			start_time = time.time()
			for step in range(num_step):
				self.average_reset()
				self.train(generate_batch(step, batch_list)[:2])
				if step % 10 == 0 and step > 0:
					print("Done with %d tweets for epoch %d"%(step*self.batch_size,epoch))
					print(time.time()-start_time)
					start_time = time.time()
				if step % 100 == 0 and step > 0:
					self.validate(validate,validate_id)
			self.save()

	def rank_on_batch(self, batch_list,case):
		print("Getting results")
		ident = case + str(np.random.randint(100))
		batch = convert2embedding(batch_list)
		feed_dict = {
			self.batch_input : batch[0],
		}
		query_similarity = self.session(self.batch_prob,feed_dict=feed_dict)
		sorted_queries = [i for i in sorted(enumerate(query_similarity),lambda x: x[1])]
		text_lines = []
		count = 0
		for t in sorted_queries:
			text_lines.append('%s Q0 %s %d %f %s'%(case,reverseTweetList[t[0]],count,t[1],ident))
			count += 1

self.batch_size = 50
tweetvec = tweet2vec(batch_size=batch_size,vocabulary_size=vocabulary_size)
print("Building model")
tweetvec.build_model()
print("Rolling session and init")
tweetvec.initialize()
print("Running for brown and reuters")
print("Running for Brown")
embeddingEncoder.train_on_batch(5,brownsentences)
embeddingEncoder.rank_on_batch(original_tweets, np.random.randint(1e6))
print("Running for reuters")
embeddingEncoder.train_on_batch(5, reutersentences)
print("Running for tweets")
embeddingEncoder.train_on_batch(10, tweetList)
embeddingEncoder.rank_on_batch(original_tweets, np.random.randint(1e6))