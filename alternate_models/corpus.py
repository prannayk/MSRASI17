import json
import numpy as np
import re
from nltk.tokenize import TweetTokenizer
from nltk.stem.lancaster import LancasterStemmer
import string

tknzr = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)
st = LancasterStemmer()
stoplist = set([i for i in map(lambda x: st.stem(x),stopwords.words('english'))])
punctuation = string.punctuation
printable = set(string.printable) 

query_words = ['need','require']
query_tokens = map(lambda x: st.stem(x),query_tokens)

def filter_fn(x):
	p1 = re.sub('[%s]*'%(punctuation),' ',x)
	p2 = re.sub(r'([a-zA-Z0-9]*)([A-Z])([a-z]*)',r'\1 \2\3',p1)
	y = st.stem(tknzr.tokenize(p2)).lower()
	return filter(lambda x: not x in stoplist and not x == '' and not len(t) == 1 and not 'www' in x and not 'http' in x,y)

print("Loading tweets")
f = open('../nepal.jsonl')
text = f.readlines()
corpus = []
count = 0
for line in text:
	count += 1
	if count % 1000 == 0:
		print(count)
	tweet = json.loads(line)
	corpus += filter_fn(x)

with open("./corpus.txt",mode="w") as fil:
	fil.write(' '.join(corpus))
print("Completed")