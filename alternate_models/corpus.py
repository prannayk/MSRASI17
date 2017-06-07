import json
import numpy as np
import re
from nltk.tokenize import TweetTokenizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
import string

tknzr = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)
st = LancasterStemmer()
stoplist = set([i for i in map(lambda x: st.stem(x),stopwords.words('english'))])
punctuation = string.punctuation
printable = set(string.printable) 

query_words = ['need','require']
query_tokens = map(lambda x: st.stem(x),query_words)

def filter_fn(x):
	p1 = re.sub('[%s]+'%(punctuation),' ',x)
	p1 = filter(lambda x: x in printable, p1)
	y = map(lambda x: st.stem(x).lower(), tknzr.tokenize(p1))
	return filter(lambda x: not x in stoplist and not x == '' and not len(x) == 1 and not 'www' in x and not 'http' in x,y)

print("Loading tweets")
f = open('../dataset/nepal.jsonl')
text = f.readlines()
corpus = []
count = 0
for line in text:
	count += 1
	tweet = json.loads(line)
	if count % 1000 == 0:
		print(count)
		print(filter_fn(tweet['text']))
	corpus += filter_fn(tweet['text'])

with open("./corpus.txt",mode="w") as fil:
	fil.write(' '.join(corpus))
print("Completed")
