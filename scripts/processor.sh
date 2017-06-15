count=57
for i in `seq 1 $1`; do
  count=$(echo "$count+1" | bc)
  filename="../alternate_models/wcattn/tweet_list_$count.txt"
  echo $filename
  cut -d' ' -f1,2,3,4 $filename > tmp.txt
  paste -d' ' tmp.txt ../alternate_models/wcattn/lol.txt > $filename
done
