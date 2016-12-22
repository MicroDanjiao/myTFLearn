#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys, math, getopt

import tensorflow as tf

from text_dataset import TextDataset
from visual_utils import plot_word_vector

class MyW2V(object):
    '''
        learn word vector of skip-gram using negative sampling.
        refer to: 
        https://github.com/tensorflow/tensorflow/blob/r0.12/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
    '''
    def __init__(self, corpus_file, method="skip-gram", batch_size=128, embedding_size=128, skip_window=2, 
                num_sampled=64, num_step=1000, loss_freq=200, num_per_win=2, min_cnt=2, lr=1.0):
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.skip_window = skip_window
        self.num_step = num_step
        self.num_sampled = num_sampled
        self.loss_freq = loss_freq
        self.method = method
        self.lr = lr

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

        if self.method == "cbow":
            # shape: (n_batch, n_ctx_win_size, embeddings_size)
            ctx_embed = tf.nn.embedding_lookup(self.embeddings, train_inputs)
            # shape: (n_batch, embeddings_size)
            embed = tf.reduce_mean(ctx_embed, axis=1)
        else:
            # shape: (n_batch, embeddings_size)
            embed = tf.nn.embedding_lookup(self.embeddings, train_inputs)


        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
        normalized_embeddings = self.embeddings / norm

        loss = tf.reduce_mean(tf.nn.nce_loss(self.nce_weights, self.nce_biases, embed, train_labels, self.num_sampled, self.vocab_size))
        optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(loss)

        average_loss = 0
        batch_iter = self.corpus.gen_skipgram_batch_iter()
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
                        average_loss /= self.loss_freq
                    print "average loss at step:", step, ":", average_loss
                    average_loss = 0
            self.final_embeddings = normalized_embeddings.eval()

        self.corpus.close()

    def save_vector(self, filename):
        with open(filename, "w") as wf:
            for i in xrange(len(self.corpus.id2word)):
                print >> wf, self.corpus.id2word[i], ' '.join(self.final_embeddings[i].astype('str'))
    
def usage():
    print "\t-h / --help"
    print "\t--corpus: corpus_file (required))"
    print "\t--method skip-gram/cbow"
    print "\t--outfile: word vector output"
    print "\t--visual: word vector picture"

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hco:v:", ["help", "corpus=", "outfile=", "visual=", "method="])
    except getopt.GetoptError:
        print "%s usage:" % sys.argv[0]
        usage()
        sys.exit(1)

    corpus = None
    outfilename = None
    picfilename = None
    method = "skip-gram"
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print "%s usage:" % sys.argv[0]
            usage()
            sys.exit(1)
        elif opt in ["method"]:
            if arg not in ["skip-gram", "cbow"]:
                usage()
                sys.exit(1)
            arg = opt
        elif opt in ("-c", "--corpus"):
            corpus = arg
        elif opt in ["-o", "--outfile"]:
            outfilename = arg
        elif opt in ["-v", "--visual"]:
            picfilename = arg
        
    if not (corpus):
        print "required arguments should be set!"
        usage()
        sys.exit(1)

    w2v = MyW2V(method=method, corpus_file=corpus, batch_size=128, embedding_size=50, skip_window=3,
                num_sampled=4, num_step=20000, loss_freq=1000, num_per_win=4, min_cnt=1, lr=1.0)

    w2v.fit()
    if outfilename is not None:
        w2v.save_vector(outfilename)

    if picfilename is not None:
        plot_word_vector(w2v.final_embeddings, w2v.corpus.id2word, pic_file=picfilename, plot_num=200)
