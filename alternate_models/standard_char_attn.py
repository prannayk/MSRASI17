from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import operator
import collections
import math
import os
import random
import zipfile
import time
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
buffer_index = 0
# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with open(filename,mode="r") as f:
    data = f.read()
    data_chars = list(set(data))
  return data.split(),data_chars,data
filename = './nepal/corpus.txt'
words,chars,character_data = read_data(filename)
print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000
with open("./nepal/data.npy") as fil:
  t = fil.readlines()
word_max_len, char_max_len = map(lambda x: int(x),t)

def build_dataset(words, vocabulary_size):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

char_dictionary = dict()
for char in chars:
  char_dictionary[char] = len(char_dictionary)

reverse_char_dictionary = dict(zip(char_dictionary.values(),char_dictionary.keys()))
char_data = []
for char in character_data:
  char_data.append(char_dictionary[char])

data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0
char_data_index = 0

# loading tweet list in integer marking form
word_batch_list = np.load("./nepal/word_embedding.npy")
char_batch_list = np.load("./nepal/char_embedding.npy")
with open("./nepal/tweet_ids.txt") as fil:
  tweet_list = map(lambda y: filter(lambda x: x != '\n',y), fil.readlines())
batch_list = dict()

# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

def generate_batch_char(batch_size, num_skips, skip_window):
  global char_data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(char_data[char_data_index])
    char_data_index = (char_data_index + 1) % len(char_data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(char_data[char_data_index])
    char_data_index = (char_data_index + 1) % len(char_data)
  # Backtrack a little bit to avoid skipping words in the end of a batch
  char_data_index = (char_data_index + len(char_data) - span) % len(char_data)
  return batch, labels

def generate_batch_train(batch_size, num_skips, skip_window):
  global buffer_index
  train_data_index = 0
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  batch_chars = np.ndarray(shape=(batch_size, char_max_len),dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  l = batch_size // word_max_len
  word_data = np.ndarray(shape=[l*word_max_len])
  char_data = np.ndarray(shape=[l*word_max_len,char_max_len])
  for i in range(l):
   word_data[word_max_len*i:word_max_len*(i+1)] = word_batch_list[buffer_index]
   char_data[word_max_len*i:word_max_len*(i+1)] = char_batch_list[buffer_index]
   buffer_index = buffer_index + 1 % len(word_batch_list)
  buffer = collections.deque(maxlen=span)
  buffer_ = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(word_data[train_data_index])
    buffer_.append(char_data[train_data_index])
    train_data_index = (train_data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      batch_chars[i*num_skips + j] = buffer_[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(word_data[train_data_index])
    buffer_.append(char_data[train_data_index])
    train_data_index = (train_data_index + 1) % len(data)
  # Backtrack a little bit to avoid skipping words in the end of a batch
  train_data_index = (train_data_index + len(word_data) - span) % len(word_data)
  return batch, batch_chars, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
batch, labels = generate_batch_char(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_char_dictionary[batch[i]],
        '->', labels[i, 0], reverse_char_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 2       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
skip_char_window = 2
num_char_skips = 3
char_vocabulary_size = len(char_dictionary)
print(char_vocabulary_size)
# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_char_size = 10
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_char_window = 20
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
valid_char_examples = np.random.choice(valid_char_window, valid_char_size, replace=False)
valid_examples[0] = dictionary['nee']
num_sampled = 64    # Number of negative examples to sample.
char_batch_size = 64
query_tokens = map(lambda x: dictionary[x],['nee','requir'])
tweet_batch_size = 10
lambda_1 = 0.7
# word_max_len
# char_max_lens

graph = tf.Graph()
learning_rate = 5e-1
with graph.as_default():

  # Input data.
  with tf.device('/gpu:0'):
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_input_chars = tf.placeholder(tf.int32, shape=[char_batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    train_char_labels = tf.placeholder(tf.int32, shape=[char_batch_size, 1])
    word_char_embeddings = tf.placeholder(tf.int32, shape=[batch_size,char_max_len])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    valid_char_dataset = tf.constant(valid_char_examples, dtype=tf.int32)
    query_ints = tf.constant(query_tokens, dtype=tf.int32)
    # Ops and variables pinned to the CPU because of missing GPU implementation
    tweet_char_holder = tf.placeholder(tf.int32, shape=[tweet_batch_size,word_max_len,char_max_len])
    tweet_word_holder = tf.placeholder(tf.int32, shape=[tweet_batch_size, word_max_len])
    # Look up embeddings for inputs.
    char_embeddings = tf.Variable(tf.random_uniform([char_vocabulary_size, embedding_size],-1.0,1.0))
    char_embed = tf.nn.embedding_lookup(char_embeddings,train_input_chars)
    lambda_2 = tf.Variable(tf.random_normal([1],stddev=1.0))

    w1 = tf.Variable(tf.random_normal([embedding_size ,embedding_size // 4],stddev=1.0/math.sqrt(embedding_size)))
    w2 = tf.Variable(tf.random_normal([embedding_size // 4,1],stddev=1.0/math.sqrt(embedding_size)))
    weights = tf.stack([w1]*batch_size)
    vvector = tf.stack([w2]*batch_size)
    weights_tweet = tf.stack([w1]*tweet_batch_size*word_max_len)
    vvector_tweet = tf.stack([w2]*tweet_batch_size*word_max_len)
    # Construct the variables for the NCE loss
    nce_char_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size ],
                            stddev=1.0 / math.sqrt(embedding_size )))
    nce_char_biases = tf.Variable(tf.zeros([vocabulary_size]))

    nce_train_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_train_biases = tf.Variable(tf.zeros([vocabulary_size]))
    
  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
    

    loss_char = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_char_weights,
                       biases=nce_char_biases,
                       labels=train_char_labels,
                       inputs=char_embed,
                       num_sampled=10,
                       num_classes=vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer_char = tf.train.AdamOptimizer(learning_rate /5).minimize(loss_char)

    # Compute the cosine similarity between minibatch examples and all embeddings.

    norm_char = tf.sqrt(tf.reduce_sum(tf.square(char_embeddings), 1, keep_dims=True))
    normalized_char_embeddings = char_embeddings / norm_char
    valid_embeddings_char = tf.nn.embedding_lookup(
        normalized_char_embeddings, valid_char_dataset)
    similarity_char = tf.matmul(
        valid_embeddings_char, normalized_char_embeddings, transpose_b=True)

    intermediate = tf.nn.embedding_lookup(normalized_char_embeddings, word_char_embeddings)
    attention = tf.nn.softmax(tf.matmul(vvector, tf.nn.tanh(tf.matmul(intermediate,weights)),transpose_a=True))
    output = tf.reshape(tf.matmul(attention,intermediate),shape=[batch_size,embedding_size])

    word_embeddings = tf.nn.embedding_lookup(normalized_embeddings, train_inputs)
    final_embedding = lambda_2*word_embeddings + (1-lambda_2)*output
    loss_char_train = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_train_weights,
                     biases=nce_train_biases,
                     labels=train_labels,
                     inputs=final_embedding,
                     num_sampled=64,
                     num_classes=vocabulary_size))

    optimizer_train = tf.train.AdamOptimizer(learning_rate/5).minimize(loss_char_train)

    tweet_word_embed = tf.nn.embedding_lookup(normalized_embeddings, tweet_word_holder)
    intermediate = tf.reshape(tf.nn.embedding_lookup(normalized_char_embeddings, tweet_char_holder),shape=[tweet_batch_size*word_max_len, char_max_len, embedding_size])
    attention = tf.nn.softmax(tf.matmul(vvector_tweet, tf.nn.tanh(tf.matmul(intermediate,weights_tweet)),transpose_a=True))
    tweet_char_embed = tf.reshape(tf.matmul(attention,intermediate),shape=[tweet_batch_size,word_max_len,embedding_size])
    tweet_embedding = tf.reduce_mean(lambda_1*tweet_word_embed + (1-lambda_1)*tweet_char_embed,axis=1)
    query_embedding = tf.reshape(tf.reduce_mean(tf.nn.embedding_lookup(normalized_embeddings,query_tokens),axis=0),shape=[1,embedding_size])
    query_similarity = tf.reshape(tf.matmul(tweet_embedding, query_embedding, transpose_b=True),shape=[tweet_batch_size])

    init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 500001
# num_steps = 0
num_steps_train = 500001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  count = 0
  print("Initialized")

  average_loss = 0
  average_char_loss = 0
  for step in xrange(num_steps):
    batch_char_inputs, batch_char_labels = generate_batch_char(
        char_batch_size, num_skips, skip_window)
    feed_dict_char = {train_input_chars: batch_char_inputs, train_char_labels: batch_char_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_char_val = session.run([optimizer_char, loss_char], feed_dict=feed_dict_char)
    average_char_loss += loss_char_val

    if step % 2000 == 0:
      if step > 0:
        print(time.time()- start_time)
        start_time = time.time()
        average_loss /= 2000
        average_char_loss /= 2000
      else:
        start_time = time.time()
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      print("Average character loss at step ", step, ": ", average_char_loss)
      average_loss = 0
      average_char_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim_char = similarity_char.eval()
      for i in xrange(valid_char_size):
        valid_word = reverse_char_dictionary[valid_char_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim_char[i, :]).argsort()[1:top_k + 1]
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = reverse_char_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
      tweet_embedding_val = []
      for t in range(len(word_batch_list) // tweet_batch_size):
        feed_dict = {
          tweet_word_holder : word_batch_list[t*tweet_batch_size:t*tweet_batch_size + tweet_batch_size],
          tweet_char_holder : char_batch_list[t*tweet_batch_size:t*tweet_batch_size + tweet_batch_size]
        }
        l = session.run(query_similarity, feed_dict = feed_dict)
        if len(tweet_embedding_val) % 1000 == 0 :
          print(len(tweet_embedding_val))
        tweet_embedding_val += list(l) 
      tweet_embedding_dict = dict(zip(tweet_list, tweet_embedding_val))
      sorted_tweets = [i for i in sorted(tweet_embedding_dict.items(), key=lambda x: -x[1])]
      for t in sorted_tweets[:100]:
        print(t[0])
      count += 1
      file_list = []
      for i in range(len(sorted_tweets)):
        file_list.append('Nepal-Need 0 %s %d %f running'%(sorted_tweets[i][0],i+1,sorted_tweets[i][1]))
      with open("./wcattn/tweet_list_%d.txt"%(count),mode="w") as fw:
        fw.write('\n'.join(map(lambda x: str(x),file_list)))
  average_loss = 0
  for step in xrange(num_steps_train):
    final_embeddings = normalized_embeddings.eval()
    final_char_embedding = normalized_char_embeddings.eval()
    np.save('./wordcharattn/word.npy',final_embeddings)
    np.save('./wordcharattn/char.npy',final_char_embedding)
    if step % 100 == 0 and step > 0:
      print(step)
    batch_inputs, batch_char_inputs, batch_labels = generate_batch_train(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, word_char_embeddings : batch_char_inputs, train_labels: batch_labels,}

    _, loss_train_val = session.run([optimizer_train, loss_char_train], feed_dict=feed_dict)
    average_loss += loss_train_val

    if step % 2000 == 0:
      if step > 0:
        print(time.time()- start_time)
        start_time = time.time()
        average_loss /= 2000
      else:
        start_time = time.time()
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0
      average_char_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      sim_char = similarity_char.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
      for i in xrange(valid_char_size):
        valid_word = reverse_char_dictionary[valid_char_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim_char[i, :]).argsort()[1:top_k + 1]
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = reverse_char_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
      tweet_embedding_val = []
      for t in range(len(word_batch_list) // tweet_batch_size):
        feed_dict = {
          tweet_word_holder : word_batch_list[t*tweet_batch_size:t*tweet_batch_size + tweet_batch_size],
          tweet_char_holder : char_batch_list[t*tweet_batch_size:t*tweet_batch_size + tweet_batch_size]
        }
        l = session.run(query_similarity, feed_dict = feed_dict)
        if len(tweet_embedding_val) % 1000 == 0 :
          print(len(tweet_embedding_val))
        tweet_embedding_val += list(l) 
      tweet_embedding_dict = dict(zip(tweet_list, tweet_embedding_val))
      sorted_tweets = [i for i in sorted(tweet_embedding_dict.items(), key=lambda x: -x[1])]
      count += 1
      file_list = []
      for i in range(len(sorted_tweets)):
        file_list.append('Nepal-Need 0 %s %d %f running'%(sorted_tweets[i][0],i+1,sorted_tweets[i][1]))
      with open("./wcattn/tweet_list_%d.txt"%(count),mode="w") as fw:
        fw.write('\n'.join(map(lambda x: str(x),file_list)))

  final_embeddings = normalized_embeddings.eval()
  final_char_embedding = normalized_char_embeddings.eval()
  weight1 = w1.eval()
  weight2 = w2.eval()
  np.save('./std_char_attn/word.npy',final_embeddings)
  np.save('./std_char_attn/char.npy',final_char_embedding)
  np.save('./std_char_attn/weight1.npy',weight1)
  np.save('./std_char_attn/weight2.npy',weight2)
