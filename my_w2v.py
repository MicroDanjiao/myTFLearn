#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys, math

import tensorflow as tf

class MyW2V(object):
    def __init__(self, vocab_size, batch_size=128, embedding_size=128, skip_window=1, num_sampled=64, 
                num_step=1000, loss_freq=200):
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.skip_window = skip_window
        self.vocab_size = vocab_size
        self.num_step = num_step
        self.num_sampled = num_sampled
        self.loss_freq = loss_freq

    def _init_param(self):
        self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
        self.nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size], stddev=1.0/math.sqrt(embedding_size)))
        self.nce_biases = tf.Variable(tf.zeros([self.vocab_size]))
    
    def fit(self.):
        self._init_param()
        init = tf.global_variable_initializer()
        train_inputs = tf.placeholder(tf.int32, shape=([self.batch_size]))
        train_labels = tf.placeholder(tf.int32, shape=([self.batch_size, 1]))

        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings)), 1, keep_dims=True)
        normalized_embeddings = self.embeddings / norm

        loss = tf.reduce_mean(tf.nn.nce_loss(nce_weight, nce_biases, embed, train_labels, self.num_sampled, self.vocab_size))
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        with tf.Session() as session:
            init.run()
            print("init params OK ...")
            for step in xrange(self.num_step):
                batch_inputs, batch_labels = generate_batch()
                feed_dict = {train_inputs: train_inputs, train_labels: batch_labels}
                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % self.loss_freq == 0:
                    if step > 0:
                        average_loss /= 200
                    print("Average loss at step:", step, ":", average_loss)
                    average_loss = 0
        self.final_embeddings = normalized_embeddings.eval()

    def transform():
        pass
