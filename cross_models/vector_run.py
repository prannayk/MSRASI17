import numpy as np
import re
from gensim.models.keyedvectors import KeyedVectors as kv
import collections
import string
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

with open("/media/hdd/hdd/data_backup/glove.twitter.27B.25d.txt") as f:
	lines = f.readlines()
wv = dict()
punctuation = string.punctuation
printable = set(string.printable)
def process(x):
	p1 = re.sub('[%s]+'%(punctuation),' ',x)
	p2 = filter(lambda x: x in printable, p1)
	return x
	return stemmer.stem(x)

for line in lines : 
	t = line.split()
	if len(wv) % 10000 == 0 :
		print(len(wv))
	wv[process(t[0])] = map(lambda x: float(x), t[1:])

dhish = 0
def build_dataset(words, vocabulary_size):
  count = [['unk', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  print(len(dictionary))
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  print(len(reverse_dictionary))
  return data, count, dictionary, reverse_dictionary
print("loaded vectors")
with open("../data/italy/corpus.txt") as f:
	t = f.readlines()
	t =map(lambda y : filter(lambda x: x!='\n', y), t)
	text = ' '.join(t)
print(len(text))
data, count, dictionary, reverse_dictionary = build_dataset(text.split(), 100000)

word_data = np.load("../data/italy/word_embedding.npy")
with open("../data/italy/tweet_ids.txt") as f:
	ids = f.readlines()
	ids = map(lambda y: filter(lambda x : x!='\n',y),ids)
word_tweet_list = ids
del ids
avail_query = np.mean(np.array([wv["give"], wv["distributed"], wv["avail"] ]),axis=0)
avail_query = avail_query / np.sqrt(np.sum(avail_query**2))
need_query = np.mean(np.array([wv["require"], wv["need"]]),axis=0)
need_query = need_query / np.sqrt(np.sum(need_query**2))
with open("../data/italy/revdict.npy") as f:
	t = f.readlines()
	t = map(lambda x: filter(lambda y: y!='\n', x), t)
reverse_dictionary = {}
for line in t:
	num = line.split(" ")[0]
	word = line.split(" ")[-1]
	reverse_dictionary[int(num)] = str(word)
reverse_dictionary[0] = "unk"
count_nf = 0
count_tot = 0
def wv_map(num):
	global count_tot, count_nf
	count_tot+=1
	word = reverse_dictionary[num]
	if word in wv :
		return wv[word]
	else :
		count_nf+=1
		print(word)
		return wv["unk"]

word_tweet_dict = {}
need_dict = {}
avail_dict = {}
for i,tweet in enumerate(word_tweet_list):
	t = np.mean(map(lambda x : wv_map(x), list(word_data[i])), axis=0)
	t = t / np.sqrt(np.sum(t**2))
	need_dict[tweet] = np.sum(t*need_query)
	avail_dict[tweet] = np.sum(t*avail_query)

need_sorted = [i for i in sorted(need_dict.items(), key=lambda x : -x[1])]
avail_sorted = [i for i in sorted(avail_dict.items(), key=lambda x: -x[1])]
file_list = []
for i in range(len(need_sorted)):
	file_list.append("Italy-Need 0 %s %d %f running"%(need_sorted[i][0],i+1, need_sorted[i][1]))
file_data = '\n'.join(file_list)
with open("eval_need.txt", mode="w") as f:
	f.write(file_data)
file_list = []
for i in range(len(avail_sorted)):
	file_list.append("Italy-Avail 0 %s %d %f running"%(need_sorted[i][0],i+1, avail_sorted[i][1]))
file_data = '\n'.join(file_list)
with open("eval_avail.txt", mode="w") as f:
	f.write(file_data)
print(count_tot)
print(count_nf)
