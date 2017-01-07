#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys, math, getopt

import tensorflow as tf

from text_dataset import TextDataset

class MyW2V(object):
    '''
        Implementation of PV-DM (Distributed Memory Model of Paragraph Vectors)
        see Distributd Representation of Sentences and Documents. Quoc le et al.
    '''
    def __init__(self, corpus_file, batch_size, embedding_size, window=2, 
                num_sampled=64, num_step=1000, loss_freq=200, min_cnt=2, lr=1.0, project="concat"):
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.window = window
        self.num_step = num_step
        self.num_sampled = num_sampled
        self.loss_freq = loss_freq
        self.lr = lr
        # projection method
        self.project = project

        # corpus iteration
        self.corpus = TextDataset(corpus_file, min_cnt=min_cnt, batch_size=batch_size, win_size=window)
        self.vocab_size = len(self.corpus.word2id)
        self.doc_size = self.corpus.num_of_lines
        print "init vocabulary size is", self.vocab_size
        print "init doc size is", self.doc_size

    def _init_param(self):
        self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
        self.doc_embeddings = tf.Variable(tf.random_uniform([self.doc_size, self.embedding_size], -1.0, 1.0))
    
        if self.project == "concat":
            # project of concatenate
            self.nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, (2*self.window+1)*self.embedding_size], stddev=1.0/math.sqrt(self.embedding_size)))
        else:
            # projection of average
            self.nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size], stddev=1.0/math.sqrt(self.embedding_size)))
        self.nce_biases = tf.Variable(tf.zeros([self.vocab_size]))
    
    def fit(self):
        self._init_param()
        init = tf.global_variables_initializer()

        train_inputs = tf.placeholder(tf.int32, shape=([self.batch_size, 2*self.window]))
        train_labels = tf.placeholder(tf.int32, shape=([self.batch_size, 1]))
        doc_ids = tf.placeholder(tf.int32, shape=([self.batch_size, 1]))

        # shape: (n_batch, 2*self.window, embeddings_size)
        ctx_embed = tf.nn.embedding_lookup(self.embeddings, train_inputs)
        # shape: (n_batch, 1, embeddings_size)
        doc_embed = tf.nn.embedding_lookup(self.doc_embeddings, doc_ids)
        print ctx_embed.get_shape()
        print doc_embed.get_shape()

        # shape (n_batch, 2*self.window+1, embeddings_size)
        concat_embed = tf.concat(1, [ctx_embed, doc_embed]) 

        if self.project == "concat":
            # shape: (n_batch, (2*self.window+1) * embeddings_size )
            embed = tf.reshape(concat_embed, [self.batch_size, -1])
        else:
            # shape: (n_batch, embeddings_size)
            embed = tf.reduce_mean(concat_embed, axis=1)
        print embed.get_shape()

        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
        normalized_embeddings = self.embeddings / norm

        loss = tf.reduce_mean(tf.nn.nce_loss(self.nce_weights, self.nce_biases, embed, train_labels, self.num_sampled, self.vocab_size))
        optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(loss)

        average_loss = 0
        batch_iter = self.corpus.gen_cbow_batch_iter()
        with tf.Session() as session:
            #session.run(tf.initialize_all_variables())
            init.run()
            print("init params OK ...")
            for step in xrange(self.num_step):
                batch_inputs, batch_labels, batch_ids = batch_iter.next()
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels, doc_ids:batch_ids}
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
    print "\t-c/--corpus: corpus_file (required)"
    print "\t-i/--iter: the number of iteration"
    print "\t-o/--outfile: word vector output"
    print "\t-v/--visual: word vector picture"

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hc:o:v:i:m:", ["help", "corpus=", "outfile=", "visual=", "method=", "iter="])
    except getopt.GetoptError:
        print "%s usage:" % sys.argv[0]
        usage()
        sys.exit(1)

    corpus = None
    outfilename = None
    picfilename = None
    method = "skip-gram"
    iter_num = 20000
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print "%s usage:" % sys.argv[0]
            usage()
            sys.exit(1)
        elif opt in ["-m", "--method"]:
            if arg not in ["skip-gram", "cbow"]:
                usage()
                sys.exit(1)
            method = arg
        elif opt in ("-c", "--corpus"):
            corpus = arg
        elif opt in ["-o", "--outfile"]:
            outfilename = arg
        elif opt in ["-v", "--visual"]:
            picfilename = arg
        elif opt in ["-i", "--iter"]:
            iter_num = int(arg)
    if not (corpus):
        print "required arguments should be set!"
        usage()
        sys.exit(1)

    print "init word2vec with %s and negative sampling" %(method)
    w2v = MyW2V(corpus_file=corpus, batch_size=128, embedding_size=50, window=3,
                num_sampled=4, num_step=iter_num, loss_freq=1000, min_cnt=1, lr=1.0)

    w2v.fit()
    if outfilename is not None:
        w2v.save_vector(outfilename)

