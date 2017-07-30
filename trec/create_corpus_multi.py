import json
import collections
import string
import time
import numpy as np
import re
import sys
from nltk.tokenize import TweetTokenizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords

tknzr = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)
st = LancasterStemmer()
stoplist = set([i for i in map(lambda x: st.stem(x),stopwords.words('english') + stopwords.words('italian') + stopwords.words('french') + stopwords.words('german'))])
punctuation = string.punctuation
printable = set(string.printable) 

query_words = ['need','require']
query_words += ['send','distribut','avail']
query_tokens = map(lambda x: st.stem(x),query_words) 

char_max_len = 0

def filter_fn(x):
	global char_max_len
	p1 = re.sub('[%s]+'%(punctuation),' ',x)
	p2 = filter(lambda x: x in printable, p1)
	y = map(lambda x: st.stem(x).lower(), tknzr.tokenize(p2))
	final = filter(lambda x: not x in stoplist and not x == '' and not len(x) == 1 and not 'www' in x and not 'http' in x,y)
	for token in final:
		if len(token) > char_max_len:
			char_max_len = len(token)
	return final

print("Loading tweets")
f = open('/media/hdd/hdd/data_backup/tweets_dict.txt')
text = f.readlines()
corpus = dict()
count = 0
word_max_len = 0
for line in text:
	count += 1
	if count % 10000 == 0:
		print(count)
	if len(line.split("\t")) > 1:
		tweet = {
			'id' : line.split("\t")[0],
			'text' : line.split("\t")[1]
		}
		corpus[tweet['id']] = filter_fn(tweet['text'])
		if len(corpus[tweet['id']]) > word_max_len:
			word_max_len = len(corpus[tweet['id']])
file = ' '.join(map(lambda x: ' '.join(x) ,corpus.values()))
with open('/media/hdd/hdd/data_backup/trec/corpus.txt',mode="w") as fil:
	fil.write(file)
def file
file = '\n'.join(map(lambda x,y : "%s\t%s"%(x,y), zip(corpus.keys(), corpus.values())))
with open("/media/hdd/hdd/data_backup/trec/marked_corpus.txt") as fil:
    fil.write(file)
del file
print("Completed tasks at hand")
exit 0
print("Written corpus to file")
words = file.split()
chars = list(set(file)) + ['.']
print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 10000000


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


print(char_dictionary['.'])

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
						train_chars[count, t , index] = char_dictionary['.']
					else:
						train_chars[count,t,index] = char_dictionary[tokens[t][index]]
				for index in range(len(tokens[t]), char_max_len):
					train_chars[count,t,index] = char_dictionary[' ']
		count += 1
	return train_word, train_chars

word_max_len = min(20,word_max_len)
char_max_len = min(20, char_max_len)
print(char_max_len)
print(word_max_len)
word_list = np.ndarray(shape=[total_size, word_max_len],dtype=np.int32)
char_list = np.ndarray(shape=[total_size, word_max_len, char_max_len],dtype=np.int32)
i=0
while i < len(corpus):
	if i % 10000 == 0 and i > 0:
		print(i)
		print(time.time() - start_time)
	start_time = time.time()
	word_markers,char_markers = convert2embedding(corpus.values()[i:i+100])
	word_list[i:i+100] = word_markers[:100]
	char_list[i:i+100] = char_markers[:100]
	i+=100
np.save('/media/hdd/hdd/data_backup/trec/word_embedding.npy',word_list)
np.save('/meda/hdd/hdd/data_backup/trec/char_embedding.npy',char_list)
l = map(lambda x: str(x), corpus.keys())
with open("/media/hdd/hdd/data_backup/trec/tweet_ids.txt",mode="w") as fil:
	fil.write('\n'.join(l))
print_list = [str(word_max_len),str(char_max_len)]
with open('/media/hdd/hdd/data_backup/trec/data.npy',mode="w") as fil:
	fil.write('\n'.join(print_list))
