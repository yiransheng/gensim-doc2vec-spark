import numpy as np
from numpy import sqrt, exp, dot, zeros, outer, random, dtype, get_include, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis, ndarray, empty, sum as np_sum
from gensim.models.word2vec import Vocab, Word2Vec
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

from operator import add
from collections import defaultdict

class DistDoc2Vec:
    '''
    DBOW, Skip-gram doc2vec model on Spark
    '''
    def __init__(self, model, alpha=0.025,
                 num_iterations=100,
                 num_partitions=None,
                 learn_hidden=True, learn_words=False):
        self.model = model # gensim model
        self.alpha = alpha # learning rate
        self.learn_hidden = learn_hidden
        self.learn_words = learn_words
        self.num_iterations = num_iterations
        self.num_partitions = num_partitions

    def build_vocab_from_rdd(self, corpus, reset_hidden=True):
        '''
        Build model vocab from RDD, respect model's min_count, max_vocab_size
        if reset_hidden sets to True (default), reset syn1neg weights
        code borrowed from:
        https://github.com/dirkneumann/deepdist/blob/master/examples/word2vec_adagrad.py
        '''
        model = self.model
        model.corpus_count = corpus.count()
        s = corpus   \
            .flatMap(lambda s: [(w, 1) for w in s])   \
            .reduceByKey(add)            \
            .filter(lambda x: x[1] >= model.min_count)              \
            .collect()
            # .map(lambda x: (x[1], x[0]))              \
            # .sortByKey(False)                         \
            # .collect()

        model.raw_vocab = defaultdict(int, s)
        model.finalize_vocab()
        model.total_words = long(len(model.vocab))

    def saveAsPickleFile(self, path):
        syn0_path = "%s.syn0" % path 
        syn1_path = "%s.syn1" % path 
        doctagsyn0_path = "%s.doctag_syn0" % path 
        self.doctag_syn0.saveAsPickleFile(doctagsyn0_path)
        sc = self.doctag_syn0.context
        sc.parallelize(self.model.syn0, 1).saveAsPickleFile(syn0_path)
        sc.parallelize(self.model.syn1, 1).saveAsPickleFile(syn1_path)

    def train_sentences_cbow(self, corpus):
        '''
        Code adaped from: https://github.com/klb3713/sentence2vec           
        to be paralleizable on Spark
        '''
        model = self.model
        alpha = self.alpha
        vocab = model.vocab
 
        def make_sent_doctag(p):
            sent, i = p
            # for now support only single-tag docuemnt/sentence
            tag = iter(sent.tags).next()
            # filter out unknown words
            words_indices = [vocab[w].index for w in sent.words \
                             if w in vocab]
            seed = "%d %s" % (model.seed, tag)
            # a single row correspoinding to Doc2Vec's doctag_syn0
            # distributed as RDD  (1xvector_size)
            docvec = model.seeded_vector(seed)
            return (words_indices, docvec, tag, i, None) # last element placeholder of syn0neg

        dataset = corpus.zipWithIndex().map(make_sent_doctag)
        if self.num_partitions:
            dataset = dataset.repartition(self.num_partitions)

        dataset = dataset.cache()

        n_part = dataset.getNumPartitions()
        sc = dataset.context

        learn_hidden = self.learn_hidden

        bc_syn0 = sc.broadcast(model.syn0) 
        bc_syn1neg = sc.broadcast(model.syn1neg) 
        bc_table = sc.broadcast(model.cum_table)
        window = model.window
        negative = model.negative
        for k in xrange(self.num_iterations):
            print "**** Training iteration % d ****" % (k+1)
            def mapPartitions(iterable):
                # sentence : [word index]
                syn1neg = bc_syn1neg.value
                syn0 = bc_syn0.value
                def result():
                    for sentence, doctag_syn0, t, i, _ in iterable:
                        for pos, w in enumerate(sentence):
                            # `b` in the original word2vec code                  
                            reduced_window = random.randint(window) 
                            start = max(0, pos - window + reduced_window)
                            window_pos = enumerate(sentence[start : pos+window+1-reduced_window],start)
                            word2_indices = [wd for pos2, wd in window_pos if pos2 != pos]
                            # layer 1
                            # uncomment to take context words as input as well
                            # l1 = np.sum(syn0[word2_indices], axis=0) / len(word2_indices)
                            # l1 += doctag_syn0
                            l1 = doctag_syn0
                            neu1e = zeros(l1.shape, dtype=np.float32)
                            word_indices = [w] 
                            table = bc_table.value
                            while len(word_indices) < negative + 1:
                                w2 = table.searchsorted(random.randint(table[-1]))
                                if w2 != w and w2 not in sentence:
                                    word_indices.append(w2) 

                            l2b = syn1neg[word_indices] # 2d matrix, neg+1 x layer1_size
                            labels = np.zeros(l2b.shape[0], dtype=REAL)
                            labels[0] = 1.0 
                            fb = 1.0 / (1.0 + exp(-dot(l1, l2b.T))) # feed forward
                            gb = (labels - fb) * alpha / sqrt(k+1) # k'th iteration
                            neu1e += dot(gb, l2b)
                            if learn_hidden:
                                delta_syn1neg = outer(gb / n_part, l1)
                                syn1neg[word_indices] += delta_syn1neg
                            doctag_syn0 += neu1e

                        yield (sentence, doctag_syn0, t, i, (word_indices, delta_syn1neg) if learn_hidden else None)
                return result()

            _dataset = dataset
            dataset = dataset.mapPartitions(mapPartitions).cache() 
            def seq_op(syn1neg, delta):
                if not delta:
                    return syn1neg
                indices, update = delta
                syn1neg[indices] += delta
                return syn1neg
            def comb_op(syn1neg_a, syn1neg_b):
                if learn_hidden:
                   syn1neg_a += syn1neg_b
                return syn1neg_a
            # 4: (indices, syn1neg deltas) or None
            if learn_hidden:
                syn1neg = dataset.map(lambda tp: tp[4]) \
                   .aggregate(np.zeros(bc_syn1neg.value.shape, dtype='float32'), 
                              seq_op, comb_op) # this triggers an action 
                bc_syn1neg.unpersist()
                model.syn1neg += syn1neg / n_part
                bc_syn1neg = sc.broadcast(model.syn1neg)
            else:
                dataset.count()
            _dataset.unpersist()

        self.doctag_syn0 = dataset.map(lambda tp: (tp[2], tp[1])) # tag, docvec
        


