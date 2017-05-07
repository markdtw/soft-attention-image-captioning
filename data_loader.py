from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import pdb
import json
import time
import random
import cPickle

import numpy as np

class Data_loader:
    # Before using data loader, make sure your data/ folder contains files generated from utils.py
    def __init__(self, batch_size=256, bias_init=False, inception=False, train=True):
        self.batch_size = batch_size
        self.bias_init = bias_init

        dictionary = cPickle.load(open('data/coco_dictionary.pkl', 'rb'))
        self.maxlen = dictionary['maxlen']
        self.wordtoix = dictionary['wordtoix']
        self.ixtoword = dictionary['ixtoword']
        self.num_words = len(dictionary['wordtoix'])
        self.biivector = dictionary['bias_init_vector'] if bias_init else None

        if not train:
            self.num_batches = 0
        else:
            print ('Loading captions and dictionary...')
            self.captions = json.load(open('data/coco_final.json', 'r'))
            self.num_examples = len(self.captions)

            print ('Loading image features...')
            ext_from = 'vgg' if not inception else 'inceptionv4'
            train_features = np.load('data/train2014_'+ext_from+'.npy')
            val_features = np.load('data/val2014_'+ext_from+'.npy')
            self.features = np.concatenate([train_features, val_features])
            del train_features, val_features

            self.create_batches()
        print ('Loading done')

    def create_batches(self):
        self.reset_batch_ptr()
        self.num_batches = int(self.num_examples / self.batch_size)
    def next_batch(self):
        if self.batch_ptr + self.batch_size >= self.num_examples: self.reset_batch_ptr()
        ctx_batch = []
        cap_batch = []
        mask_batch = []
        for i in xrange(self.batch_size):
            ctx_batch.append(self.features[self.captions[i + self.batch_ptr]['image_ix']])
            cap_batch.append(self.captions[i + self.batch_ptr]['caption'])
            mask_batch.append(self.captions[i + self.batch_ptr]['mask'])
        self.batch_ptr += self.batch_size
        return np.asarray(ctx_batch), cap_batch, mask_batch
    def reset_batch_ptr(self):
        self.batch_ptr = 0
        random.shuffle(self.captions)
