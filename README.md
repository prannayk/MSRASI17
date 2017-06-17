# MSRASI17
Work done for the Artificial Social Intelligence 2017, Microsoft Research

## Main steps:
*
* Deep network based retrieval for ``` need ``` and ``` avail ``` tweets 
* Information extraction using NLP + vector representation techniques
* Matching of said tweets

## System dependencies:
* Python >= 2.7
* Bazel latest
* CUDA >= 7.5
* CUDNN >= 5.1 (>= 6.0 soon)
* Tensorflow >= 1.1.0
* other dependencies related to above

The retrieval is being done on 2 datasets : ``` Nepal earthquake ``` , ``` Italy earthquake ```

## Implemented Models:
* Character Level embeddings
* Word and Character Level embeddings in skipgram setting
* Word embedding with attention over character embedding
* Word embedding with attention over BiLSTM character embedding

## Running types:
* No query expansion
* Query expansion

Mode switiching is unimplemented and for now is being done by changes to source code

## Models:
* CLE : Character Level embeddings that are trained using Character Level context
* WC1 : Word and Character Level embeddings that are both combined together and trained to predict the context of the token ``` skipgram ``` method
* WC2 : Word and Character Level embeddings that are combined after applying ``` attention ``` to character sequence of the token while training is done in ``` skipgram ``` setting
* WC3 : Word embeddings and attention over Character level ``` BiLSTM ``` model for token embedding extraction while training is done in ``` skipgram ``` setting

The evaluation is run with ``` ./trec eval -q -m <measure standards> <standard> <output> ```

The data is available / was available under ``` FIRE2016 ```

#### These codes are part of a research project and will remain private till released publicly. When released they will be available under MIT license and therefore free for anyone to use till the time the work is cited by whoever who uses it. 

## Utility documentation

Mostly all utility methods are present in different files which perfectly define the use of the function. The transfer of variables was benchmarked to observe the slowdown did not exist. 

## Information for model creation
* The models must have 4 specific placeholders, apart / including those made for training
* They are namely ``` tweet_query_char_holder ``` , ``` tweet_query_word_holder ``` , ``` tweet_word_holder ``` , ``` tweet_char_holder ``` 
* Tweet holders take batch input for tweets which are to be evaluated
* Tweet query holder take input the tweet which is to be used as query
* There must be atleast a single ``` tweet_similarity ``` tensor along the model which computes the the required metric for tweet retrieval acording to which sorting shall happen
* Library does not support multiple testing metrics, but will be implemented shortly

#### The code was written solely by Prannay Khosla during Workshop conducted by Microsoft Research India on Artificial Social Intelligence. The work was done under Prof. Saptarshi Ghosh, Assistant Professor IIT Kanpur. 

#### The workshop was conducted by Microsoft Research India, Bangalore, #9 Lavelle Road

NOTE : For access to datasets please contact ``` prannay[dot]khosla[at]gmail[dot]com ```


