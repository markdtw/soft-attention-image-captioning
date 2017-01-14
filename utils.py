from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from random import shuffle
import cPickle
import time
import pdb
import re
import os

import pandas as pd
import numpy as np

class Data_loader:

    def __init__(self, params, inference=False):
        self.params = params
        captions = cPickle.load(open(params.data_dir+'train_captions_ultimate.pkl', 'rb'))
        self.captions, sentences = self.captionProcessing(captions)
        self.captions = pd.DataFrame(self.captions)
        
        maxlen, word_to_index, index_to_word, bias_init_vector = self.preProBuildWordVocab(sentences, params.word_thrh)
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.bivec = bias_init_vector
        self.n_words = len(word_to_index)
        self.maxlen = maxlen

        if inference:
            if params.eval_all:
                #TODO
                pass
            else:
                self.img_path = params.img_path
        else:
            self.imgs_vgg = np.load(params.data_dir+'train_vggc54npf16.npy')
            print ('Features loaded.')
            self.create_batches()
    
    def convert_idx_mask(self, sequs_batch):
        new_seq = []
        masks = []
        for seq in sequs_batch:
            sent_list = seq.split(' ')
            masks.append([1]*(len(sent_list)+1) + [0]*(self.maxlen-len(sent_list)))
            # masks done
            sent_idx = []
            for word in sent_list:
                if word in self.word_to_index:
                    sent_idx.append(self.word_to_index[word])
                else:
                    sent_idx.append(self.word_to_index['#RARE#'])
            sent_idx += [0]*(self.maxlen - len(sent_idx) + 1)
            new_seq.append(sent_idx)
            # new_seqs done
        return new_seq, masks

    def next_batch(self):
        if self.batch_pointer + self.params.batch_size >= len(self.captions):
            self.reset_batch_pointer()
        
        sequs_batch = self.captions['caption'][self.batch_pointer:self.batch_pointer+self.params.batch_size]
        sequs_batch, masks_batch = self.convert_idx_mask(sequs_batch)
        sequs_batch = np.asarray(sequs_batch)
        masks_batch = np.asarray(masks_batch)

        feats_index = self.captions['image_index'][self.batch_pointer:self.batch_pointer+self.params.batch_size]
        feats_batch = self.imgs_vgg[feats_index]
        feats_batch = feats_batch.reshape(-1, self.params.ctx_shape[0], self.params.ctx_shape[1])
        
        self.batch_pointer += self.params.batch_size

        return feats_batch, sequs_batch, masks_batch
    def reset_batch_pointer(self):
        self.batch_pointer = 0
        self.captions = self.captions.sample(frac=1).reset_index(drop=True)
    def create_batches(self):
        self.reset_batch_pointer()
        self.num_batches = int(len(self.captions) / self.params.batch_size)

    def captionProcessing(self, captions):
        """Remove
        1. all the non-alphanumerical characters
        2. multiple white spaces
        3. leading or trailing white spaces
        Convert all to lower case
        """
        sentences = []
        for c in captions:
            c['caption'] = re.sub(r'[^\w^\s]', '', c['caption'])
            c['caption'] = re.sub(r'\s+', ' ', c['caption'])
            c['caption'] = c['caption'].strip().lower()
            sentences.append(c['caption'])
        return captions, sentences
    
    def preProBuildWordVocab(self, sentence_iterator, word_count_threshold=100):
        """This function is from karpathy/neuraltalk with a slight modification.
        Additionally, counting the max length of a sentence.
        Args:
            sentence_iterator: all the captions
            word_count_threshold: word count below this number will be treated as RARE, TA gave us 100

        Return:
            maxlen: longest caption length
            wordtoix: dec_map by TA
            ixtoword: enc_map by TA
            bias_init_vector: the legend said that it will reduce the 'perplexity'
        """
        # count up all word counts so that we can threshold
        # this shouldnt be too expensive of an operation
        print ('Preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, ))
        t0 = time.time()
        word_counts = {}
        nsents = 0
        maxlen = 0
        for sent in sentence_iterator:
            nsents += 1
            sent_list = sent.split(' ')
            if len(sent_list) > maxlen:
                maxlen = len(sent_list)
            for w in sent_list:
                word_counts[w] = word_counts.get(w, 0) + 1
        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
        print ('Filtered words from %d to %d in %.2fs' % (len(word_counts), len(vocab), time.time() - t0))
    
        # with K distinct words:
        # - there are K+1 possible inputs (START token and all the words)
        # - there are K+1 possible outputs (END token and all the words)
        # we use ixtoword to take predicted indeces and map them to words for output visualization
        # we use wordtoix to take raw words and get their index in word vector matrix
        ixtoword = {}
        ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
        ixtoword[1] = '#RARE#'
        wordtoix = {}
        wordtoix['#START#'] = 0 # make first vector be the start token
        wordtoix['#RARE#'] = 1
        ix = 2
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1
        
        # compute bias vector, which is related to the log probability of the distribution
        # of the labels (words) and how often they occur. We will use this vector to initialize
        # the decoder weights, so that the loss function doesnt show a huge increase in performance
        # very quickly (which is just the network learning this anyway, for the most part). This makes
        # the visualizations of the cost function nicer because it doesn't look like a hockey stick.
        # for example on Flickr8K, doing this brings down initial perplexity from ~2500 to ~170.
        word_counts['.'] = nsents
        word_counts['#RARE#'] = word_count_threshold
        bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
        bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
        bias_init_vector = np.log(bias_init_vector)
        bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
        return maxlen, wordtoix, ixtoword, bias_init_vector
