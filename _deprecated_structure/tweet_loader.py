import json 
import string
printable = list(set(string.printable) - set('\n'))
f = open("./nepal.jsonl")

text = f.readlines()
tweetList = list()
file_lines = []
count = 0
for line in text:
	print(count)
	count += 1
	tweet = json.loads(line)
	file_lines.append(''.join([i for i in filter(lambda x: x in printable,'%s\t%s'%(tweet['id'],tweet['text']))]))

with open("./tweet_list.txt",mode="w") as f:
	f.write('\n'.join(file_lines))