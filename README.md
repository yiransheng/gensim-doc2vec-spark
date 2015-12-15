# Gensim Doc2Vec on Spark

## Overview

Implements a prototype for running `gensim` `Doc2Vec` on Spark. Only `PV-DBOW` with negative sampling is implemented.
This work is inspired by:

https://github.com/dirkneumann/deepdist

https://github.com/klb3713/sentence2ve

## General Idea

Most ML models in `Spark`'s `MLLib` only palatalizes training process, while keeping model parameters in driver program and broadcast to workers. This works for models which themselves are small enough to hold in memory on a single node, while training set can be large and has to be parallelized as `RDD`s. However, `Doc2Vec` models are not of this category - number of model parameters is linear to number of points in dataset. 

The goal of `Doc2Vec` is to learn vector representation of _each_ document in training set. For example, a dataset of 10 million documents and vector size 300, requires 300,000,000 floating number parameters (or a 300x1000,000 array). Fortunately, each data point (a.ka. sentence or document) only updates its corresponding row in the weights matrix during training process, therefore, it's possible to parallelize the model by zipping its parameters with training dataset: each partition only holds the parameters relevant to its own share of data. 

`gensim` is used as a basis for this setup, training for sentence vectors are adapted to work on `RDD`s. 

## Details

When training `Doc2Vec` in `PV-DBOW` model and using negative sampling, three `numpy` arrays are of interests in `gensim`, the fully captures the model state:

1. `model.syn0`
2. `model.syn1neg`
3. `model.docvecs.doctag_syn0`

In our implementation, we keep `syn0` and `syn1neg` centralized, as they are of limited size (size of total vocabulary). `doctag_syn0` is held as RDD (each partition holds a single `numpy` array for it). `Word2Vec` model is broadcasted to all partitions. 

In each training iteration, the following happens:

1. On each partition, `Cython` and `BLAS` powered procedure `train_document_dbow` from `gensim.models.doc2vec_inner` is called, and trains word vectors, document vector and hidden layer weights jointly
2. We record triplet (`syn0` deltas, `syn1neg` deltas, `doctag_syn0`) and produce a new RDD with each partition holding a single triplet; and this new RDD is cached (as it will be used twice)
3. We aggregate all deltas through Spark's `RDD.aggregate` api, to sum all deltas, then apply deltas to `model` object in driver program
4. Previous generation of model broadcased is unpersisted, new model is broadcasted to all executors (runs actual training as `aggregate` is a Spark action)
5. Create new inputs from new RDD in step 4, training will not be re-run as we have cached results, then we invalid stale triplet RDD from previous iteration

By tweaking `num_partitions` and `num_iterations`, we can balance the trade-off between accuracy, speed and network overhead. 

## Test Results

Cornell Movie Reviews [Dataset](http://www.cs.cornell.edu/people/pabo/movie-review-data/) is used to test the approach out. Model is trained on 5 partitions and 20 iterations, and we were able to classify movie reviews labels with about **11%** error rate only from docvectors, with balanced false negative and postive rate: 

```
*** Error Rate: 0.107995 ***
*** False Positive Rate: 0.107799 ***
*** False Negative Rate: 0.108191 ***
```



