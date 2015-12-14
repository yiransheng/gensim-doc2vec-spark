from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
 
import numpy as np


conf = (SparkConf() \
    .set("spark.driver.maxResultSize", "4g"))

sc = SparkContext(conf=conf)
# sqlContext = SQLContext(sc)
pos = sc.textFile("hdfs:///movie_review/positive").map(lambda s: (True, s.lower().split()))
neg = sc.textFile("hdfs:///movie_review/negative").map(lambda s: (False, s.lower().split()))

if False:
    docvecs = sc.pickleFile("hdfs://movie_review/doctags")
else:
    from ddoc2vecf import DistDoc2VecFast

    data = (neg + pos).zipWithIndex().map(lambda (v, i): (i, v[0], v[1]))
    sents = data.map(lambda (a,b,c): c)

    model = Word2Vec(size=100, hs=0, negative=8)
    dd2v = DistDoc2VecFast(model, learn_hidden=False, num_partitions=5, num_iterations=3)
    dd2v.build_vocab_from_rdd(sents, reset_hidden=False)
    # train word2vec in driver 
    model.train(sents.collect())
    model.save("/root/doc2vec/word2vec_model/review")
    print "*** done training words ****"
    print "*** len(model.vocab): %d ****" % len(model.vocab)
    dd2v.train_sentences_cbow(data.map(lambda (i, l, v): TaggedDocument(words=v, tags=[i])))
    # dd2v.saveAsPickleFile("hdfs:///movie_review/docvectors")
    def split_vec(iterable):
        dvecs = iter(iterable).next()
        n = np.shape(dvecs)[0]
        return (dvecs[i] for i in xrange(n))

    docvecs = dd2v.doctag_syn0.mapPartitions(split_vec)
        

npos = pos.count()
reg_data = docvecs.zipWithIndex().map(lambda (v, i): LabeledPoint(1.0 if i<npos else 0.0, v))
(trainingData, testData) = reg_data.randomSplit([0.7, 0.3])

lrmodel = LogisticRegressionWithLBFGS.train(trainingData)
labelsAndPreds = testData.map(lambda p: (p.label, lrmodel.predict(p.features)))

trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(testData.count())
falsePos = labelsAndPreds.filter(lambda (v, p): v != p and v == 0.0).count() / float(testData.filter(lambda lp: lp.label == 0.0).count())
falseNeg = labelsAndPreds.filter(lambda (v, p): v != p and v == 1.0).count() / float(testData.filter(lambda lp: lp.label == 1.0).count())

print "*** Error Rate: %f ***" % trainErr
print "*** False Positive Rate: %f ***" % falsePos
print "*** False Negative Rate: %f ***" % falseNeg




