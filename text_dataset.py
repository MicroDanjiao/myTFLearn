#!/user/bin/env python
# -*- coding:utf-8 -*-

import os, sys, collections
import numpy as np

pre_lst = ["\"", "\'"]
post_lst = [".", ",", "!", "?", ";", "\"", "\'"]

def strip_word(word):
    if len(word) <= 1:
        return word
    ori_word = word
    if word[0] in pre_lst:
        word = word[1:]
    if word[-1] in post_lst:
        word = word[:-1]
    if len(word) == 0:
        return ori_word
    return word

class TextDataset(object):
    '''
        this class used to generate dataset for w2v
    '''
    def __init__(self, filename, min_cnt=2, batch_size=128, win_size=1, num_per_win=win_size):
        if not os.path.isfile(filename):
            raise Exception("the file %s not found" % filename)

        self.min_cnt = min_cnt
        self.filename = filename
        self.batch_size = batch_size
        self.win_size = win_size
        self.num_per_win = num_per_win
        assert num_per_win <= 2 * win_size
    
        self.rf = open(filename, "r")
        _build_dataset()

    def _build_dataset(self):
        counter = collections.Counter()
        word2id = {"UNK", 0}
        self.rf.seek(0, 0):
        for line in rf.readline():
            words = [strip_word(w.lower()) for w in line.strip().split()]
            counter.update(words)

        print "vocabulary size: %d" %len(counter)
        unk_cnt = 0
        rm_word = []
        for key, value in counter.iteritems
            if value <= self.miin_cnt:
                unk_cnt += value
                rm_word.append(key)
            else:
                word2id[key] = len(word2id)

        for w in rm_word:
            del counter[w]
        counter["UNK"] = unk_cnt

        id2word = dict(zip(word2id.values(), word2id.keys()))
        self.word2id = word2id
        self.id2word = id2word
        self.wordcnt = counter

    def generate_batch(self):
        self.rf.seek(0,0)
        batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        target = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        while True:
            line = self.rf.readline()
            if line == "":
                self.rf.seek(0,0)
                line = self.rf.readline()
            words = [strip_word(w.lower()) for w in line.strip().split()]
            if len(words) <= 1:
                continue
            wids = [self.word2id.get(w, 0) for w in words]
            for i in range(len(wids)):
                target = wids[i]
                 context = 
                
                

    
    def close():
        self.rf.close()
