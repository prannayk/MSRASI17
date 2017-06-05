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
character_window = 2
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
#print("Loading Brown corpus")
#brownsentences = sentence_processor([i for i in brown.sents()])
#len_brown_sents = len(brownsentences)
#print("Loading Reuters corpus")
#reutersentences = sentence_processor([i for i in reuters.sents()])
#len_reuters_sents = len(reutersentences)
#print("Loading Twitter corpus")
tweetList = sentence_processor(tweetList)
original_tweets = list(tweetList)
#tweetList += sentence_processor([i for i in twitter_samples.strings()])
print("Loaded everything")
print("Read and processed tweets and tokens")
tokenList = process_tweets(tweetList, 1e-7)
print("Done with tweetList")	
#browntokens = token_processor(brownsentences)
#reutertokens = token_processor(reutersentences)
print("Merging: ")
browntokens = []
reutertokens = []
tokenList = list(set(browntokens + reutertokens + tokenList.keys() + query_tokens) - set(stoplist))
print("Processing tokens")
tokenList = map(lambda x: re.sub('[%s]*'%(punctuation),'',x), filter(lambda x: filter_fn(x) ,tokenList))
#brownsentences = map(lambda y: filter(lambda x: filter_fn(x),y), brownsentences)
#reutersentences = map(lambda y: filter(lambda x: filter_fn(x),y), reutersentences)
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
	return train_word, train_chars

class attention_char():
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
		self.num_index = 0
		# variables
		with tf.device("/cpu:00"):
			self.char_embeddings = tf.Variable(tf.random_normal(shape=[char_size, char_embedding_size],stddev=1.0))
			self.word_embeddings = tf.Variable(tf.random_normal(shape=[vocabulary_size, word_embedding_size], stddev=1.0))
			# attention matrix
			self.weight1 = tf.Variable(tf.random_normal(shape=[char_embedding_size,self.dim1]))
			self.weight2 = tf.Variable(tf.random_normal(shape=[self.dim1,self.dim2]))
			self.weight3 = tf.Variable(tf.random_normal(shape=[self.dim2,self.dim3]))


	def embedding_creator(self,train_chars, train_words,flag=False):
		with tf.device("/cpu:0"):
			self.weights1 = tf.stack([[self.weight1]*word_max_len]*batch_size)
			self.weights2 = tf.stack([[self.weight2]*word_max_len]*batch_size)
			self.weights3 = tf.stack([[self.weight3]*word_max_len]*batch_size)
			self.ir_weight = {
				'weight1' : tf.stack([[self.weight1]*self.word_max_len]*self.num_queries),
				'weight2' : tf.stack([[self.weight2]*self.word_max_len]*self.num_queries),
				'weight3' : tf.stack([[self.weight3]*self.word_max_len]*self.num_queries)
			}
			words = tf.nn.embedding_lookup(self.word_embeddings,train_words)
			chars = tf.nn.embedding_lookup(self.char_embeddings,train_chars)
			if not flag:
				attention1 = tf.sigmoid(batch_normalize(tf.matmul(chars,self.weights1)))
				attention2 = tf.sigmoid(batch_normalize(tf.matmul(attention1,self.weights2)))
				attention3 = tf.sigmoid(batch_normalize(tf.matmul(attention2,self.weights3)))
				hidden_layer = tf.matmul(attention3, chars, transpose_a = True)
				character_embedding = tf.reshape(hidden_layer,shape=[self.batch_size, self.word_max_len, self.char_embedding_size])
			else:
				attention1 = tf.sigmoid(batch_normalize(tf.matmul(chars,self.ir_weight['weight1'])))
				attention2 = tf.sigmoid(batch_normalize(tf.matmul(attention1,self.ir_weight['weight2'])))
				attention3 = tf.sigmoid(batch_normalize(tf.matmul(attention2,self.ir_weight['weight3'])))
				hidden_layer = tf.matmul(attention3, chars, transpose_a = True)
				character_embedding = tf.reshape(hidden_layer,shape=[self.num_queries, self.word_max_len, self.char_embedding_size])
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
			words_matrix = tf.reshape(tf.transpose(normalized_embeddings_word), shape=[1,1,self.word_embedding_size,self.vocabulary_size])
			similarity = tf.nn.conv2d(valid_run, words_matrix, padding='SAME', strides = [1,1,1,1])

			self.saver = tf.train.Saver()
			self.init = tf.global_variables_initializer()
			self.validwords = valid_words
			self.v_chars = valid_chars
			self.optimizer = optimizer
			self.loss = loss
			self.similarity = similarity

			self.ir_words = tf.placeholder(tf.int32,shape=[self.batch_size, self.word_max_len])
			self.ir_chars = tf.placeholder(tf.int32, shape=[self.batch_size, self.word_max_len, self.char_max_len])

			ir_embedding = self.embedding_creator(self.ir_chars,self.ir_words)
			valid_ir = tf.reduce_mean(ir_embedding,axis=1)
			self.query_lit.append(tf.placeholder(tf.int32,shape=[self.num_queries, self.word_max_len,self.char_max_len]))
			self.query_lit.append(tf.placeholder(tf.int32,shape=[self.num_queries, self.word_max_len]))
			query_vectors = tf.reduce_mean(self.embedding_creator(self.query_lit[0],self.query_lit[1],flag=True),axis=1)
			self.query_similarity = tf.reduce_max(tf.matmul(query_vectors,valid_ir,transpose_b=True),axis=0)

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
	def train(self,batch):
		self.index += 1
		feed_dict = {
			self.train_words : batch[0],
			self.train_chars : batch[1]
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
		with open("./last_attention_run.text",mode="w") as fil:
			fil.write(filedata.encode('utf-8','ignore'))

	def train_on_batch(self,num_epoch, batch_list):
		num_step = len(batch_list) // self.batch_size
		validate = generate_batch(np.random.randint(1,20),batch_list)
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

	def create_query(self,num_queries,query_tokens,num_total):
		print("Petrol")
		self.num_queries = num_queries
		self.query_tokens = query_tokens
		query_size = len(query_tokens)
		self.total_batch_size = num_total
		query_tweet_list = []
		for r in range(num_queries):
			l,t = np.random.randint(query_size,size=[2])
			while l == t:
				l,t = np.random.randint(query_size,size=[2])
			query_tweet_list.append([self.query_tokens[l],self.query_tokens[t]])
		self.query_list = convert2embedding(query_tweet_list)
		self.query_lit = list()

	def rank_on_batch(self, batch_list,case):
		print("Getting results")
		ident = str(case) + str(np.random.randint(100))
		query_similarity = []
		for i in range(int(math.ceil(len(batch_size) / self.batch_size))):
			batch = convert2embedding(batch_list[i*self.batch_size : i*self.batch_size + batch_size])
			feed_dict = {
				self.ir_words : batch[0],
				self.ir_chars : batch[1],
				self.query_lit[0] : self.query_list[1],
				self.query_lit[1] : self.query_list[0]
			}
			query_similarity += self.session.run(self.query_similarity,feed_dict=feed_dict)
		sorted_queries = [i for i in sorted(enumerate(query_similarity),key=lambda x: -x[1])]
		text_lines = []
		count = 0
		for t in sorted_queries:
			text_lines.append('%s Q0 %s %d %f %s'%(case,reverseListing[t[0]],count,t[1],ident))
			count += 1
		with open('./char_cbow.result.text',mode="w") as f:
			f.write('\n'.join(text_lines))


print("Entering Embedding maker")
attention_model = attention_char(
		learning_rate = 5e-1,
		dim1 = 64, dim2=16, dim3=1, 
		char_embedding_size = 128,
		word_embedding_size = 128,
		char_max_len = char_max_len,
		word_max_len = word_max_len,
		batch_size = batch_size,
		char_size = char_size,
		vocabulary_size = vocabulary_size,
		beta = 0.001,
		valid_words = None,
		valid_chars = None
	)

attention_model.create_query(5,query_tokens,len(original_tweets))
print("Building model")
_ = attention_model.build_model()
print("Setting up session")
session = attention_model.session()
print("Running init")
attention_model.initialize()
print("Variables Initialized")
print("Running for brown and reuters")
print("Running for Brown")
#attention_model.train_on_batch(5,brownsentences)
print("Running for reuters")
#attention_model.train_on_batch(5, reutersentences)
print("Running for tweets")
attention_model.train_on_batch(10, tweetList)
attention_model.rank_on_batch(original_tweets, np.random.randint(1e6))
