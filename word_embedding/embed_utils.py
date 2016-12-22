#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import numpy as np

class EmbedUtils(object):
    '''
        utils to insight the word embeddings
    '''
    def __init__(self, embed_file):
        self.id2word = []
        embeddings = []
        
        # load word embedding file
        with open(embed_file, "r") as rf:
            for line in rf:
                tokens = rf.readlines().split()
                self.id2word.append(tokens[0])
            embed = [int(w) for w in tokens[1:]]    
            embeddings.append(embed)

        self.embeddings = np.array(embeddings) 
        self.word2id = {}
        for i, w in enumerate(self.id2word):
            self.word2id[w] = i

    def analogy(self, w0, w1, w2, n_top=5):
        '''
            predict word w4 with analogy w0:w1 vs w2:w3
        '''
        for w in (w0, w1, w2):
            if w not in self.word2id:
                print "%s is not in the vocabulary" %w
                return []
        a_id = self.word2id[w1]
        b_id = self.word2id[w2]
        c_id = self.word2id[w3]
        a_emb = self.embeddings[a_id]
        b_emb = self.embeddings[b_id]
        c_emb = self.embeddings[c_id]
        
        target = c_emb + (b_emb - a_emb)
        
        dist = np.matmul(self.embeddings, target).ravel()
        top_k = dist.argsort(dist)[len(dist)-1:len(dist)-1-n_top:-1]
        res = []
        for k in top_k:
            res.append((k, dist[k]))
        return res

    def nearby(self, word, n_top=10):
        '''
            predict nearby words given a word
        '''
        if word not in self.word2id:
            print "%s is not in the vocabulary" %word
            return []
        wid = self.word2id[word]
        emb = self.embeddings[wid]
        dist = np.matmul(self.embeddings, emb).ravel()
        top_k = dist.argsort(dist)[len(dist)-1:len(dist)-1-n_top:-1]
        res = []
        for k in top_k:
            res.append((k, dist[k]))
        return res
    

if __name__ == "__main__":
    pass
