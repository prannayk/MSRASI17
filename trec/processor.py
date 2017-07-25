import json
import os
from os import listdir
from os.path import isfile, join

path = "/media/hdd/hdd/data_backup/tweet.terrier.format"
files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
print(files)
tweet_list = {}
flag=False
count = 0
for name in files:
	with open(name) as f:
		text = f.readlines()
	bui=False
	for line in text:
		if line.startswith("<tweettime>"):
			last_time = filter(lambda x : x!='\n', line.split(">")[1].split("<")[0])
		elif "<TEXT>" in line:
			flag = True
		elif flag:
			tweet_list[last_time] = line
			flag = False
			bui = True
		count += 1	
		if count % 100000 == 0 and bui:	
			bui=False
			# print(line)
			print(count)
		bui=False
	print(name + " %d"%(count))

with open("tweet_list.txt", mode="w") as f:
	f.write('\n'.join(map(lambda  (x,y) : "%d\t%s"%(x,y), zip(tweet_list.keys(), tweet_list.values()))))

print("Done")
