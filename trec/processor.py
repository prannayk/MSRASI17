import json
import os
from os import listdir
from os.path import isfile, join

path = "/media/hdd/hdd/data_backup/"
files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

tweet_list = {}
for name in files:
	with open(name) as f:
		text = f.readlines()
	for line in text:
		if line.startswith("<tweettime>"):
			last_time = int(line.split(">")[1].split("<")[0])
		elif line == "<TEXT>"
			flag = True
		elif flag:
			tweet_list[last_time] = line
			flag = False

with open("tweet_list.txt", mode="w") as f:
	f.write('\n':join(map(lambda  (x,y) : "%d\t%s"%(x,y), zip(tweet_list.keys(), tweet_list.placeholders()))))

print("Done")