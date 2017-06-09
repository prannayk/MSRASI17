std=$(ls ../alternate_models/stdwcattn -la | grep -c "tweet")
wc=$(ls ../alternate_models/wcattn -la | grep -c "tweet")
char=$(ls ../alternate_models/char2vec -la | grep -c "tweet")
pchar=$(ls ../alternate_models/plain_char2vec -la | grep -c "tweet")

echo "Standard word char attention"
./run $std stdwcattn | tail -n30 | grep "[0-9].[0-9]*: "
echo "Wore char attention LSTM"
./run $wc wcattn | tail -n30 | grep "[0-9]*.[0-9]*"
echo "char2vec"
./run $char char2vec | tail -n30 | grep "[0-9]*.[0-9]*"
echo "plain char2vec"
./run $pchar plain_char2vec | tail -n30 | grep "[0-9]*.[0-9]*"
