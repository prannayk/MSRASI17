import json
import time
import numpy as np
import re
from nltk.tokenize import TweetTokenizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords

tknzr = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)
st = LancasterStemmer()
stoplist = set([i for i in map(lambda x: st.stem(x),stopwords.words('english'))])
punctuation = string.punctuation
printable = set(string.printable) 

query_words = ['need','require']
query_tokens = map(lambda x: st.stem(x),query_words)

def filter_fn(x):
	p1 = re.sub('[%s]+'%(punctuation),' ',x)
	p2 = filter(lambda x: x in printable, p1)
	y = map(lambda x: st.stem(x).lower(), tknzr.tokenize(p2))
	return filter(lambda x: not x in stoplist and not x == '' and not len(x) == 1 and not 'www' in x and not 'http' in x,y)

print("Loading tweets")
f = open('../dataset/nepal.jsonl')
text = f.readlines()
corpus = []
count = 0
word_max_len = 0
for line in text:
	count += 1
	if count % 1000 == 0:
		print(count)
	tweet = json.loads(line)
	corpus[tweet['id']] = filter_fn(tweet)
	if len(corpus[tweet['id']]) > word_max_len:
		word_max_len = len(corpus[tweet['id']])

def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with open(filename,mode="r") as f:
    data = f.read()
    data_chars = list(set(data))
  return data.split(),data_chars,data
filename = './corpus.txt'
words,chars,character_data = read_data(filename)
print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000


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

data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

total_size = len(corpus)


# Step 4: Build and train a skip-gram model.
def convert2embedding(batch):
	global char_dictionary, dictionary
	global char_max_len, word_max_len
	train_word = np.ndarray([len(batch),word_max_len],dtype=np.int32)
	train_chars = np.ndarray([len(batch),word_max_len, char_max_len])
	count = 0
	for tweet in batch:
		tokens = tweet
		for t in range(word_max_len):
			if t >= len(tokens):
				train_word[count, t] = dictionary['UNK']
				train_chars[count, t] = np.zeros_like(train_chars[count,t])
			else:
				if tokens[t] in dictionary:
					train_word[count, t] = dictionary[tokens[t]]
				else:
					train_word[count, t] = dictionary['UNK']
				for index in range(min(char_max_len, len(tokens[t]))):
					if tokens[t][index] in punctuation:
						train_chars[count, t , index] = char_dictionary['-']
					else:
						train_chars[count,t,index] = char_dictionary[tokens[t][index]]
				for index in range(len(tokens[t]), char_max_len):
					train_chars[count,t,index] = char_dictionary[' ']
		count += 1
		if count % 100 == 0 and count > 0:
			print(count)
	return train_word, train_chars

word_list = np.ndarray(shape=[total_size, word_max_len],dtype=np.int32)
char_list = np.ndarray(shape=[total_size, word_max_len, char_max_len],dtype=np.int32)

for i in range(len(corpus)):
	word_list[i:i+1],char_list[i:i+1] = convert2embedding(corpus[corpus.keys()[i:i+1]])
embeddings = np.concat([word_list.reshape(shape=[total_size, 1, word_max_len]),word_list.reshape(shape=[total_size,1,word_max_len,char_max_len])],axis=1)

np.save('processed_tweets.npy',embeddings)

with open("./tweet_ids.txt") as fil:
	fil.write('\n'.join(corpus.keys()))
