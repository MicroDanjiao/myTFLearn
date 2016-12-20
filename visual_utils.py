#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys

from sklearn.manifold import TSNE
import matplotlib

def plot_word_vector(embeddings, id2word, low_dim=2, pic_file="tsne.png", plot_num=500):
    '''
        plot word vector
    '''
    assert plot_num <= len(id2word)
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tsne = TSNE(perplexity=30, n_components=low_dim, init='pca', n_iter=5000)
    low_dim_embs = tsne.fit_transform(embeddings[:plot_num, :])
    labels = [id2word[i] for i in xrange(plot_num)]
    
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x,y), xytext=(5,2), textcoords='offset points', ha='right', va='bottom')
    plt.savefig(pic_file)
    
