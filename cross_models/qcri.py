import numpy as np
from gensim.models.keyedvectors import KeyedVectors as kv
import collections
wv = kv.load_word2vec_format("/media/hdd/hdd/data_backup/crisisNLP_word2vec_model/crisisNLP_word_vector.bin", binary="True")
dhish = 0
def build_dataset(words, vocabulary_size):
  global dhish
  count = [['unk', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    if word == "unk" : 
      dhish = len(dictionary) + 1
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
with open("../data/nepal/corpus.txt") as f:
	t = f.readlines()
	t =map(lambda y : filter(lambda x: x!='\n', y), t)
	text = ' '.join(t)
print(len(text))
data, count, dictionary, reverse_dictionary = build_dataset(text.split(), 100000)

word_data = np.load("../data/nepal/word_embedding.npy")
with open("../data/nepal/tweet_ids.txt") as f:
	ids = f.readlines()
	ids = map(lambda y: filter(lambda x : x!='\n',y),ids)
word_tweet_list = ids
del ids
avail_query = np.mean(np.array([wv["giv"], wv["distribut"], wv["avail"] ]),axis=0)
avail_query = avail_query / np.sqrt(np.sum(avail_query**2))
need_query = np.mean(np.array([wv["requir"], wv["need"]]),axis=0)
need_query = need_query / np.sqrt(np.sum(need_query**2))
reverse_dictionary[0] = "unk"
def wv_map(num):
	if num in reverse_dictionary :
		word = reverse_dictionary[num]
	else :
		print(num)
		word = reverse_dictionary[dhish]
	if word in wv :
		return wv[word]
	else :
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
	file_list.append("Nepal-Need 0 %s %d %f running"%(need_sorted[i][0],i+1, need_sorted[i][1]))
file_data = '\n'.join(file_list)
with open("eval_need.txt", mode="w") as f:
	f.write(file_data)
file_list = []
for i in range(len(avail_sorted)):
	file_list.append("Nepal-Avail 0 %s %d %f running"%(need_sorted[i][0],i+1, avail_sorted[i][1]))
file_data = '\n'.join(file_list)
with open("eval_avail.txt", mode="w") as f:
	f.write(file_data)

