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
        return strip_word(word)
    if word[-1] in post_lst:
        word = word[:-1]
        return strip_word(word)
    if len(word) == 0:
        return ori_word
    return word

class TextDataset(object):
    '''
        this class used to generate dataset for w2v
    '''
    def __init__(self, filename, min_cnt=2, batch_size=128, win_size=1, num_per_win=1):
        '''
            attention: the context is (win_size target win_size),
        '''
        if not os.path.isfile(filename):
            raise Exception("the file %s not found" % filename)

        self.min_cnt = min_cnt
        self.filename = filename
        self.batch_size = batch_size
        self.win_size = win_size
        self.num_per_win = num_per_win
        assert num_per_win <= 2 * win_size
    
        self.rf = open(filename, "r")
        self._build_dataset()

    def _build_dataset(self):
        counter = collections.Counter()
        word2id = {"UNK": 0}
        self.rf.seek(0, 0)
        for line in self.rf:
            words = [strip_word(w.lower()) for w in line.strip().split()]
            counter.update(words)

        unk_cnt = 0
        rm_word = []
        for key, value in counter.iteritems():
            if value < self.min_cnt:
                unk_cnt += value
                rm_word.append(key)
            else:
                word2id[key] = len(word2id)

        for w in rm_word:
            del counter[w]
        counter["UNK"] = unk_cnt

        # "EMPTY" is placeholder of the start and end of the sentences for CBOW
        word2id["_EMPTY_"] = len(word2id)
        print "vocabulary size: %d" %len(counter)
        id2word = dict(zip(word2id.values(), word2id.keys()))
        self.word2id = word2id
        self.id2word = id2word
        self.wordcnt = counter

    def gen_skipgram_batch_iter(self):
        '''
            batch iteration for skip-gram
        '''
        self.rf.seek(0,0)
        batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        batch_idx = 0
        while True:
            line = self.rf.readline()
            if line == "":
                self.rf.seek(0,0)
                line = self.rf.readline()
            #print line
            words = [strip_word(w.lower()) for w in line.strip().split()]
            if len(words) <= 1:
                continue
            wids = [self.word2id.get(w, 0) for w in words]
            #print wids
            #[window target window]
            for i in range(len(wids)):
                center_id = wids[i]
                ctx_ids = []
                for j in range(i-self.win_size, i+self.win_size+1):
                    if j != i and 0 <= j < len(wids):
                        ctx_ids.append(wids[j])

                # shuffle the context words
                np.random.shuffle(ctx_ids)
                for ctx_id in ctx_ids[:self.num_per_win]:
                    batch[batch_idx] = center_id
                    labels[batch_idx] = ctx_id
                    batch_idx += 1
            
                    # one batch is filled
                    if batch_idx == self.batch_size:
                        yield batch, labels
                        batch_idx = 0
                        batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
                        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
   

    def gen_cbow_batch_iter(self):
        '''
            batch iteration for CBOW
            the context window size is 2*win_size
            '_EMPTY_' placeholder will fill the window if context words are not enough
        '''
        self.rf.seek(0,0)
        ctx_win_size = 2*self.win_size
        batch = np.ndarray(shape=(self.batch_size, ctx_win_size), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        batch_idx = 0
        while True:
            line = self.rf.readline()
            if line == "":
                self.rf.seek(0,0)
                line = self.rf.readline()
            #print line
            words = [strip_word(w.lower()) for w in line.strip().split()]
            
            if len(words) <= 1:
                continue
            wids = [self.word2id.get(w, 0) for w in words]
            #print wids
            #[window target window]
            for i in range(len(wids)):
                center_id = wids[i]
                ctx_ids = []
                for j in range(i-self.win_size, i+self.win_size+1):
                    if j != i and 0 <= j < len(wids):
                        ctx_ids.append(wids[j])
               
                # fill the context 
                placeholder_id = self.word2id["_EMPTY_"]
                for k in range(ctx_win_size-len(ctx_ids)):
                    ctx_ids.append(placeholder_id)

                batch[batch_idx, :] = ctx_ids
                labels[batch_idx] = center_id
                batch_idx += 1
            
                # one batch is filled
                if batch_idx == self.batch_size:
                    yield batch, labels
                    batch_idx = 0
                    batch = np.ndarray(shape=(self.batch_size, ctx_win_size), dtype=np.int32)
                    labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)                    

    def close(self):
        self.rf.close()

if __name__ == "__main__":
    td = TextDataset("data/test", min_cnt=1, batch_size=5, win_size=2, num_per_win=2)
    print td.word2id
    print td.id2word
    batch_iter = td.gen_cbow_batch_iter()
    for x in range(10):
        batch, labels = batch_iter.next()
        print "batch", len(batch)
        print batch
        print "lables", len(labels)
        print labels.ravel()
        print "********"
    td.close()

