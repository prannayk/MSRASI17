#!/bin/bash
i=0
for j in `seq 1 49`; do
	i=$(echo "$i+1" | bc)
	grep "^$i " ../data/trec/microblog11-qrels.txt | cut -d' ' -f3 > "mob$i.txt"
done
grep 