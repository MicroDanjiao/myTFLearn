#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys, math, getopt

import tensorflow as tf

from text_dataset import TextDataset

class MyW2V(object):
    def __init__(self, corpus_file, batch_size=128, embedding_size=128, skip_window=2, 
                num_sampled=64, num_step=1000, loss_freq=200, num_per_win=2, min_cnt=2):
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.skip_window = skip_window
        self.num_step = num_step
        self.num_sampled = num_sampled
        self.loss_freq = loss_freq

        # corpus iteration
        self.corpus = TextDataset(corpus_file, min_cnt=min_cnt, batch_size=batch_size, win_size=skip_window, num_per_win=num_per_win)
        self.vocab_size = len(self.corpus.word2id)
        print "init vocabulary size is", self.vocab_size

    def _init_param(self):
        self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
        self.nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size], stddev=1.0/math.sqrt(self.embedding_size)))
        self.nce_biases = tf.Variable(tf.zeros([self.vocab_size]))
    
    def fit(self):
        self._init_param()
        #init = tf.global_variables_initializer()
        train_inputs = tf.placeholder(tf.int32, shape=([self.batch_size]))
        train_labels = tf.placeholder(tf.int32, shape=([self.batch_size, 1]))

        embed = tf.nn.embedding_lookup(self.embeddings, train_inputs)

        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
        normalized_embeddings = self.embeddings / norm

        loss = tf.reduce_mean(tf.nn.nce_loss(self.nce_weights, self.nce_biases, embed, train_labels, self.num_sampled, self.vocab_size))
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        average_loss = 0
        batch_iter = self.corpus.gen_batch_iter()
        with tf.Session() as session:
            session.run(tf.initialize_all_variables())
            print("init params OK ...")
            for step in xrange(self.num_step):
                batch_inputs, batch_labels = batch_iter.next()
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % self.loss_freq == 0:
                    if step > 0:
                        average_loss /= 200
                    print "Average loss at step:", step, ":", average_loss
                    average_loss = 0
            self.final_embeddings = normalized_embeddings.eval()

        self.corpus.close()

    def transform():
        pass

def usage():
    print "\t-h / --help"
    print "\t--corpus: corpus_file (required))"
    print "\t--vocab_size: the size of vocabulary (required)"        

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["help", "corpus=", "vocab_size="])
    except getopt.GetoptError:
        print "%s usage:" % sys.argv[0]
        usage()
        sys.exit(1)

    corpus = None
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print "%s usage:" % sys.argv[0]
            usage()
            sys.exit(1)
        elif opt in ("--corpus"):
            corpus = arg
        
    if not (corpus):
        print "required arguments should be set!"
        usage()
        sys.exit(1)

    w2v = MyW2V(corpus_file=corpus, batch_size=128, embedding_size=128, skip_window=2,
                num_sampled=1, num_step=1000, loss_freq=200, num_per_win=2, min_cnt=2)

    w2v.fit()
