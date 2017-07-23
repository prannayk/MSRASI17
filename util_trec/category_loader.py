import numpy as np

dictionary = dict()
def load_categories():
	list_cats = []
	# dictionary = dict()
	global dictionary
	for i in range(48):
		with open("../data/trec/mob%d.txt"%(i+1)) as fil:
			text = fil.readlines()
			text = map(lambda x: filter(lambda y: y != '\n',x),text)
		list_cats.append(text)
		for tweet in text : 
			if tweet in dictionary:
				dictionary[tweet].append(i+1)
			else :
				dictionary[tweet] = [i+1]
	return dictionary, list_cats

def generate_pair(dictionary, list_cats, word_batch_dict, batch_size, word_max_len):
	list_common = list(set(word_batch_dict.keys()) - (set(word_batch_dict.keys()) - set(dictionary.keys())))
	print(len(list_common))
	t1 = np.random.choice(word_batch_dict.keys(), size=batch_size, replace=False)
	t2 = np.random.choice(word_batch_dict.keys(), size=batch_size, replace=False)
	tweets1 = np.zeros([2*batch_size, word_max_len])
	tweets2 = np.zeros([2*batch_size, word_max_len])
	marker = np.zeros([3*batch_size])
	for i in range(batch_size):
		set1 = set(dictionary[t1[i]])
		set2 = set(dictionary[t2[i]])
		if len(set1) == len(set1 - set2):
			marker[i] = 1
		else :
			marker[i] = 0
		tweets1[i] = word_batch_dict[t1[i]]
		tweets2[i] = word_batch_dict[t2[i]]
	t = np.randon.randint(list_cats)
	t1 = np.random.choice(list_cats[t], size=batch_size, replace=False)
	t2 = np.random.choice(list_cats[t], size=batch_size, replace=False)
	for i in range(batch_size):
		tweets1[batch_size+i] = word_batch_dict[t1[i]]
		tweets2[batch_size+i] = word_batch_dict[t2[i]]
		marker[batch_size+i] = 1
	return [tweets1,tweets2], marker
