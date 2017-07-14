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
from gensim.models import KeyedVectors

dataset = "complete"
dataset2 = "italy"

tknzr = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)
st = LancasterStemmer()
stoplist = set([i for i in map(lambda x: st.stem(x),stopwords.words('english'))])
punctuation = string.punctuation
printable = set(string.printable) 

query_words = ['need','require'] # query tokens
query_words += ['send','distribut','avail'] # query tokens
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
f = open('../dataset/%s.jsonl'%(dataset))
text = f.readlines()
corpus = dict()
corpus_file = list()
count = 0
word_max_len = 0
for line in text:
	count += 1
	if count % 1000 == 0:
		print(count)
	tweet = json.loads(line)
	corpus[tweet['id']] = filter_fn(tweet['text'])
	if len(corpus[tweet['id']]) > word_max_len:
		word_max_len = len(corpus[tweet['id']])
	corpus_file += [(tweet['id'], corpus[tweet['id']])]
tweets = dict(corpus_file)



en_model = KeyedVectors.load_word2vec_format("/media/hdd/hdd/data_backup/crisisNLP_word2vec_model/crisisNLP_word_vector.bin",binary=True)

def embed(x):
	if x in en_model:
		return en_model[x]
	else:
		return en_model['unk']

# tweet_stuff = np.array(map(lambda x: map(lambda z: embed(filter(lambda y: y!= '\n',z),x),tweets.values()))
tweet_stuff = np.array(map(lambda x: np.array(map(embed, ''.join(filter(lambda y: y!='\n',x)).split())),tweets.values()))
print(tweet_stuff.shape)
tweet_embed = np.mean(tweet_stuff,axis=1)
print(tweet_embed.shape)

query = np.mean(np.array([en_model['need'],en_model['require']]),axis=0).reshape()

similarity = np.matmul(query, np.transpose(tweet_embed))

r = dict(zip(tweets.keys(), similarity))
sorted_tweets = [i for i in sorted(tweet_embedding_dict.items(), key=lambda x: -x[1])]
file_list = []
for i in range(len(sorted_tweets)):
	dataset_name = list(dataset)
	dataset_name[0] = dataset[0].upper()
	dataset_name[1:] = dataset[1:]
	if int(sorted_tweets[i][0][0]) == 5:
		something = "Nepal"
	else:
		something = "Italy"
	file_list.append('%s-%s 0 %s %d %f running'%(something, query_name,sorted_tweets[i][0],i+1,sorted_tweets[i][1]))

with open("../results/qcri_list.txt",mode="w") as f:
	f.write('\n'.join(file_list))
