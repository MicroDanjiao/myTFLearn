#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys, math, getopt

import tensorflow as tf
import numpy as np

from text_dataset import TextDataset

class MyW2V(object):
    '''
        Implementation of PV-DM (Distributed Memory Model of Paragraph Vectors)
        see Distributd Representation of Sentences and Documents. Quoc le et al.
    '''
    def __init__(self, corpus_file, batch_size, embedding_size, window=2, num_sampled=64, num_step=1000, loss_freq=200, min_cnt=2, lr=1.0, project="concat", op="train", vector_file=None):
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.window = window
        self.num_step = num_step
        self.num_sampled = num_sampled
        self.loss_freq = loss_freq
        self.lr = lr
        # projection method
        self.project = project
        self.op = op

        # corpus iteration
        if op == "train":
            self.corpus = TextDataset(corpus_file, min_cnt=min_cnt, batch_size=batch_size, win_size=window)
        else:
            self.load_word_vector(vector_file)      
            self.corpus = TextDataset(corpus_file, min_cnt=min_cnt, batch_size=batch_size, win_size=window)
            
        self.vocab_size = len(self.corpus.word2id)
        self.doc_size = self.corpus.num_of_lines

        print "init vocabulary size is", self.vocab_size
        print "init doc size is", self.doc_size

    def _init_param(self):
        if self.op == "train":
            self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
        else:
            self.embeddings = tf.constant(self.word_embeddings, dtype=tf.float32)

        self.doc_embeddings = tf.Variable(tf.random_uniform([self.doc_size, self.embedding_size], -1.0, 1.0))
    
        if self.project == "concat":
            # project of concatenate
            self.nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, (2*self.window+1)*self.embedding_size], stddev=1.0/math.sqrt(self.embedding_size)))
        else:
            # projection of average
            self.nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size], stddev=1.0/math.sqrt(self.embedding_size)))
        self.nce_biases = tf.Variable(tf.zeros([self.vocab_size]))
        
        self.train_inputs = tf.placeholder(tf.int32, shape=([self.batch_size, 2*self.window]))
        self.train_labels = tf.placeholder(tf.int32, shape=([self.batch_size, 1]))
        self.doc_ids = tf.placeholder(tf.int32, shape=([self.batch_size, 1]))

        # shape: (n_batch, 2*self.window, embeddings_size)
        ctx_embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
        # shape: (n_batch, 1, embeddings_size)
        doc_embed = tf.nn.embedding_lookup(self.doc_embeddings, self.doc_ids)
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

        if op == "train":
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
            self.normalized_embeddings = self.embeddings / norm

        doc_norm = tf.sqrt(tf.reduce_sum(tf.square(self.doc_embeddings), 1, keep_dims=True))
        self.normalized_doc_embeddings = self.doc_embeddings / doc_norm

        self.loss = tf.reduce_mean(tf.nn.nce_loss(self.nce_weights, self.nce_biases, embed, self.train_labels, self.num_sampled, self.vocab_size))
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)    

    def fit(self):
        self._init_param()
        init = tf.global_variables_initializer()

        average_loss = 0
        batch_iter = self.corpus.gen_cbow_batch_iter()
        with tf.Session() as session:
            #session.run(tf.initialize_all_variables())
            init.run()
            print("init params OK ...")
            for step in xrange(self.num_step):
                batch_inputs, batch_labels, batch_ids = batch_iter.next()
                feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels, self.doc_ids:batch_ids}
                _, loss_val = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % self.loss_freq == 0:
                    if step > 0:
                        average_loss /= self.loss_freq
                    print "average loss at step:", step, ":", average_loss
                    average_loss = 0
            self.final_embeddings = self.normalized_embeddings.eval()
            self.final_doc_embeddings = self.normalized_doc_embeddings.eval()

        self.corpus.close()

    def transform(self):
        self._init_param()
        init = tf.global_variables_initializer()

        average_loss = 0
        batch_iter = self.corpus.gen_cbow_batch_iter()

        with tf.Session() as session:
            init.run()
            print "init params OK ..."
            for step in xrange(self.num_step):
                batch_inputs, batch_labels, batch_ids = batch_iter.next()           
                feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels, self.doc_ids:batch_ids}
                _, loss_val = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % self.loss_freq == 0:
                    if step > 0:
                        average_loss /= self.loss_freq
                    print "average loss at step:", step, ":", average_loss
                    average_loss = 0
            self.final_doc_embeddings = self.normalized_doc_embeddings.eval()
  
    def save_vector(self, filename):
        if self.op == "train":
            with open(filename+"_word", "w") as wf:
                for i in xrange(len(self.corpus.id2word)):
                    print >> wf, self.corpus.id2word[i], ' '.join(self.final_embeddings[i].astype('str'))
        with open(filename+"_doc", "w") as wf:
            for i in xrange(self.doc_size):
                print >> wf, i, ' '.join(self.final_doc_embeddings[i].astype('str'))

    def load_word_vector(self, filename):
        word_embeddings = []
        idx = 0
        with open(filename+"_word", "r") as rf:
            for line in rf:
                tokens = line.strip().split()
                word = tokens[0]
                embed = tokens[1:]
                word_embeddings.append(embed)
                idx += 1
        self.word_embeddings = np.array(word_embeddings).astype(np.float)
    
def usage():
    print "\t-h / --help"
    print "\t-t / --train_or_trans (default: train)"
    print "\t-c/--corpus: corpus_file (required)"
    print "\t-i/--iter: the number of iteration"
    print "\t-o/--outfile: word vector and doc vector output"
    print "\t-f/--infile: the file name of word vector (for trans)"

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hc:o:i:t:f:", ["help", "corpus=", "outfile=", "visual=", "iter=", "train_or_trans=", "infile"])
    except getopt.GetoptError:
        print "%s usage:" % sys.argv[0]
        usage()
        sys.exit(1)

    corpus = None
    outfilename = None
    picfilename = None
    iter_num = 20000
    op = "train"
    infilename = None
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print "%s usage:" % sys.argv[0]
            usage()
            sys.exit(1)
        elif opt in ["-t", "--train_or_trans"]:
            if arg not in ["train", "trans"]:
                usage()
                sys.exit(1)
            op = arg
        elif opt in ("-c", "--corpus"):
            corpus = arg
        elif opt in ["-o", "--outfile"]:
            outfilename = arg
        elif opt in ["-i", "--iter"]:
            iter_num = int(arg)
        elif opt in ["-f", "--infile"]:
            infilename= arg             

    if not (corpus):
        print "required arguments should be set!"
        usage()
        sys.exit(1)

    if op != "train" and infilename is None:
        usage()
        sys.exit()

    print "%s the sentence vector" % op
    #print "init word2vec with %s and negative sampling" %(method)
    w2v = MyW2V(corpus_file=corpus, batch_size=128, embedding_size=50, window=3, num_sampled=4, num_step=iter_num, loss_freq=1000, min_cnt=1, lr=1.0, op = op, vector_file=infilename)
    
    if op == "train":
        w2v.fit()
    else:
        w2v.transform()
    if outfilename is not None:
        w2v.save_vector(outfilename)

