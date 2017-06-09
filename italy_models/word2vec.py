from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile
import json
import re
import math
import time
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

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
avail_words = ['available','distribute','given','giving','sending']
avail_tokens = map(lambda x: st.stem(x).lower(),avail_words)
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
  if not tweet['id'] in reverseListing.values():
    tweetList.append(tknzr.tokenize(filter(lambda x: x in printable,tweet['text']).decode('utf-8','ignore')))
    reverseListing[count] = tweet['id']
    count += 1
print("Loaded tweets")

maxlen = 0
maxlen_upper_limit = 50
maxsize_upper_limit = 50
browntokens = []
reutertokens = []

tweetList = sentence_processor(tweetList)
original_tweets = list(tweetList)
print("Read and processed tweets and tokens")
tokenList = process_tweets(tweetList, 1e-7)
print("Done with tweetList")

tokenList = list(set(browntokens + reutertokens + tokenList.keys() + query_tokens + avail_tokens) - set(stoplist))
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
  train_word = np.ndarray([len(batch)*word_max_len],dtype=np.int32)
  train_chars = np.ndarray([len(batch)*word_max_len, char_max_len])
  count = 0
  for tweet in batch:
    tokens = tweet
    for t in range(word_max_len):
      if t >= len(tokens):
        train_word[count*word_max_len + t] = word2count['UNK']
      else:
        if tokens[t] in word2count:
          train_word[count*word_max_len + t] = word2count[tokens[t]]
        else:
          train_word[count*word_max_len + t] = word2count['UNK']
    count += 1
    if count % 1000 == 0 and count > 0:
      print(count)
  return train_word, train_chars

def generate_batch(splice,batch_list):
  slic = splice*batch_size + batch_size % len(batch_list)
  batch = batch_list[slic-batch_size:slic]
  train_word, _ = convert2embedding(batch)
  count = 0
  global word_max_len, word2count
  train_labels = np.ndarray([batch_size * word_max_len, 1])
  for tweet in batch:
    for t in range(word_max_len):
      l = t + np.random.randint(-skip_window, skip_window+1)
      while l >= word_max_len or l < 0:
        l = t + np.random.randint(-skip_window,skip_window+1)
      if l < len(tweet):
        if tweet[l] in word2count:
          train_labels[count*word_max_len + t,0] = word2count[tweet[l]]
        else:
          train_labels[count*word_max_len + t, 0] = word2count['UNK']
      else:
        train_labels[count*word_max_len + t,0] = word2count['UNK']
    count += 1  
  return train_word, train_labels

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
valid_examples[0] = word2count['need']
valid_examples[1] = word2count['requir']
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size*word_max_len])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size*word_max_len, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = len(original_tweets) // batch_size
st = 0
with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print("Initialized")

  average_loss = 0
  for roll in range(1000):
    for step in xrange(num_steps):
      batch_inputs, batch_labels = generate_batch(step, tweetList)
      feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
      _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
      average_loss += loss_val

      if st % 100 == 0:
        if step > 0:
          average_loss /= 100
      # The average loss is an estimate of the loss over the last 2000 batches.
        print("Average loss at step ", step, ": ", average_loss)
        average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
      if st % 1000 == 0:
        sim = similarity.eval()
        for i in xrange(valid_size):
          valid_word = count2word[valid_examples[i]]
          top_k = 8  # number of nearest neighbors
          nearest = (-sim[i, :]).argsort()[1:top_k + 1]
          log_str = "Nearest to %s:" % valid_word
          for k in xrange(top_k):
            close_word = count2word[nearest[k]]
            log_str = "%s %s," % (log_str, close_word)
          print(log_str)
      st+=1
  final_embeddings = normalized_embeddings.eval()
