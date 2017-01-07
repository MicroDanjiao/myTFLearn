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
        This class used to generate dataset for w2v
        The line number will be used, so one line should be one sentence or one paragraph.     
    '''
    def __init__(self, filename, min_cnt=2, batch_size=128, win_size=1):
        '''
            attention: the context is (win_size target win_size),
        '''
        if not os.path.isfile(filename):
            raise Exception("the file %s not found" % filename)

        self.min_cnt = min_cnt
        self.filename = filename
        self.batch_size = batch_size
        self.win_size = win_size
    
        self.rf = open(filename, "r")
        self._build_dataset()

    def _build_dataset(self):
        counter = collections.Counter()
        word2id = {"UNK": 0}
        self.rf.seek(0, 0)
        num_of_lines = 0    # the number of lines
        for line in self.rf:
            words = [strip_word(w.lower()) for w in line.strip().split()]
            counter.update(words)
            num_of_lines += 1

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

        # "_EMPTY_" is placeholder of the start and end of the sentences for CBOW
        word2id["_EMPTY_"] = len(word2id)
        print "vocabulary size: %d" %len(counter)
        id2word = dict(zip(word2id.values(), word2id.keys()))
        self.word2id = word2id
        self.id2word = id2word
        self.wordcnt = counter
        self.num_of_lines = num_of_lines

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
        lineids = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)

        batch_idx = 0
        line_idx = -1
        while True:
            line = self.rf.readline()
            line_idx += 1
            #print line, line_idx
            if line == "":
                self.rf.seek(0,0)
                line = self.rf.readline()
                line_idx = 0
                #print "start to pass through the text from beginning..."
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
                lineids[batch_idx] = line_idx
                batch_idx += 1
            
                # one batch is filled
                if batch_idx == self.batch_size:
                    yield batch, labels, lineids
                    batch_idx = 0
                    batch = np.ndarray(shape=(self.batch_size, ctx_win_size), dtype=np.int32)
                    labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)                    
                    lineids = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)                    

    def close(self):
        self.rf.close()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print "Usage: %s filename" %sys.argv[0]
        sys.exit()

    filename = sys.argv[1]

    td = TextDataset(filename, min_cnt=1, batch_size=5, win_size=2)
    print td.word2id
    print td.id2word
    batch_iter = td.gen_cbow_batch_iter()
    for x in range(10):
        batch, labels, lineids = batch_iter.next()
        print "batch", len(batch)
        print batch
        print "lables", len(labels)
        print labels.ravel()
        print "lineids", len(lineids)
        print lineids
        print "********"
    td.close()

