---
title: "Gensim Doc2Vec on Spark"
author: "Yiran Sheng"
date: "12/13/2015"
output: html_document
---

# Gensim Doc2Vec on Spark

## Overview

Implements a prototype for running `gensim` `Doc2Vec` on Spark. Training for document vectors are pure python (with numpy) as of now, which is 20x-80x slower than `gensim`'s optimized C version. Therefore, running this is less likely to be faster than running `gensim` on a single node. Only `PV-DBOW` with negative sampling is implemented, and works best with a pre-trained `Word2Vec` model using `hs=0` and `negative>0` settings (skip-gram, negative sampling), so we don't need to learn the weights on hidden layer (`learn-hidden=False`).

This work is inspired by:

https://github.com/dirkneumann/deepdist

https://github.com/klb3713/sentence2ve

## General Idea

Most ML models in `Spark`'s `MLLib` only palatalizes training process, while keeping model parameters in driver program and broadcast to workers. This works for models which themselves are small enough to hold in memory on a single node, while training set can be large and has to be parallelized as `RDD`s. However, `Doc2Vec` models are not of this category - number of model parameters is linear to number of points in dataset. 

The goal of `Doc2Vec` is to learn vector representation of _each_ document in training set. For example, a dataset of 10 million documents and vector size 300, requires 300,000,000 floating number parameters (or a 300x1000,000 array). Fortunately, each data point (a.ka. sentence or document) only updates its corresponding row in the weights matrix during training process, therefore, it's possible to parallelize the model by zipping its parameters with training dataset: each partition only holds the parameters relevant to its own share of data. 

`gensim` is used as a basis for this setup, training for sentence vectors are adapted to work on `RDD`s. 

## Algorithum

`PV-DBOW` [(paper)](https://cs.stanford.edu/~quocle/paragraph_vector.pdf) works very similarly to Skip Gram model in `word2vec` models. The model is forced to predict random word sampled from a sentence/document. 

> ... at each iteration of stochastic gradient descent, we sample a text window, then sample a random word from the text window and form a classifi-cation task given the Paragraph Vector.

Using negative sampling, instead of training softmax classifier, we train a simple logistic regression model. For example, giving a sentence with id `SENT_0` and context window size of 1:

```
the quick brown fox jumped over the lazy dog
```

We have context, target pairs of:

```
([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox), ...
```

And training sampels of:

```
(quick, the), (quick, brown), (brown, quick), (brown, fox), ...
```

The goal of negative sampling PV-DBOW model is to maximize the objective function, where "quick" is the target word, and "sheep" is a randomly sampled noisy word:

J(θ) = log<sub>θ</sub>[D=1|SENT_0, the] + log<sub>θ</sub>[D=0|SENT_0, sheep]

(This is similar to word2vec model, which maximizes: J(θ) = log<sub>θ</sub>[D=1|quick,the] + log<sub>θ</sub>[D=0|quick,sheep], the target word is replaced with sentence id `SENT_0`, this makes it possible to reuse hidden->output weights obtained from word2vec model training).

The implementation in `gensim` for skip-gram, negative sampling can be found in function [`train_sg_pair`](https://github.com/piskvorky/gensim/blob/develop/gensim/models/word2vec.py#L223), which is reused in `Doc2Vec` python implementation as well. 

When training a single sentence, first, it's vector form is looked up (from `model.docvecs.doctag_syn0`) using id `SENT_0`, dot multiplied with vectors for word "the" and "sheep" looked up from the hidden->output layer weights (`model.syn1neg`), then fed into the logsistic function to produce an output between 0.0 and 1.0 (feed forward). The second step is to propagate error back to the input-> hidden layer (or just update the document vector with error gradient). 

Training on a sentence will _only_ update its own vector representation (the row corresponds to its id in input-> hidden weights matrix), if we freeze the learning of hidden->output layer (`numpy` array `model.syn1neg`). 

## Moving on to a cluster

In `gensim`, document vectors are stored as `model.docvecs.doctag_syn0`, which is a `numpy` memmapped array (so that it does not blow up memory for large training dataset). This makes it impractical to broadcast the models to workers when running on Spark, as we'd have to copy the backing array (on disk) to multiple nodes. 

Assuming we have a already trained `gensim` `Word2Vec` model, we broadcast it to all workers, so that we can access helper functions on the model object anywhere. 

The first step is to parallelize this potentially very big numpy array. For a training set in RDD form of N documents and p partitions, we create a single `doctag_syn0` numpy array on each partition through `mapPartitions` operation. Subsequent training is performed on the RDD produced by zipping training set and this parameter RDD (RDD[numpy.array]). 

Training happens on each partition, for each iteration, the `numpy` array holder each partition's `doctags_syn0`is updated in place using a custom training function (a series of `numpy` array operations). Resulting parameters is collected as a new RDD through `mapPartitions`. 

## Usage

Main class `DistDoc2Vec`. Constructed by:

```
DistDoc2Vec(model=model, # gensim word2vec model
            alpha=0.025, # learning rate
            learn_hidden=False, # update syn1neg or not
            num_iterations=10,   # number of training iterations
            num_partitions=None  # number of partitions for RDD, if None will use input RDD's settings)
```

Build vocab from RDD:

```
build_vocab_from_rdd(rdd)

# rdd is a RDD of sentences, a sentence is a list of tokens/words
```

Training on Spark:

```
train_sentences_cbow(rdd)

# Rdd is a RDD of TaggedDocument (gensim) objects 
```


## Example

```
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
from ddoc2vec import DistDoc2Vec

# sents is a RDD of sentences (array of tokens)

model = Word2Vec(size=100, hs=0, negative=8)  
dd2v = DistDoc2Vec(model, learn_hidden=False, num_partitions=5, num_iterations=10)
dd2v.build_vocab_from_rdd(sents, reset_hidden=False)
# train word2vec in driver
model.train(sents.collect())
dd2v.train_sentences_cbow(sents.zipWithIndex().map(lambda (s, i): TaggedDocument(words=s, tags=[i])))
```

Running `test.py` (on movie review data found [here](http://www.cs.cornell.edu/people/pabo/movie-review-data/)):

```
$SPARK_HOME/bin/spark-submit --verbose \
   --master yarn \
   --deploy-mode client \
   --py-files ddoc2vec.py \
   ./test.py
```

