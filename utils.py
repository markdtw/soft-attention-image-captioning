from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import pdb
import os

import numpy as np

class Data_loader:

    def __init__(self, params):
        self.imgs_vgg = np.load(params.data_dir+'imgs_vggc54.npy')
        
        all_captions = cPickle.load(open(params.data_dir+'captions_train14.pkl', 'rb'))
        
        word_to_index, index_to_word, bias_init_vector = self.preProBuildWordVocab(sentences, 100)
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.n_words = len(word_to_index)



    def preProBuildWordVocab(self, sentence_iterator, word_count_threshold):
        """This function is from karpathy/neuraltalk, replacing TA's utils.
        Additionally, I counted the max length of a sentence.

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
        print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
        t0 = time.time()
        word_counts = {}
        nsents = 0
        maxlen = 0
        for sent in sentence_iterator:
            nsents += 1
            if len(sent) > maxlen:
                maxlen = len(sent)
            for w in sent['tokens']:
                word_counts[w] = word_counts.get(w, 0) + 1
        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
        print 'filtered words from %d to %d in %.2fs' % (len(word_counts), len(vocab), time.time() - t0)
    
        # with K distinct words:
        # - there are K+1 possible inputs (START token and all the words)
        # - there are K+1 possible outputs (END token and all the words)
        # we use ixtoword to take predicted indeces and map them to words for output visualization
        # we use wordtoix to take raw words and get their index in word vector matrix
        ixtoword = {}
        ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
        wordtoix = {}
        wordtoix['#START#'] = 0 # make first vector be the start token
        ix = 1
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
        bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
        bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
        bias_init_vector = np.log(bias_init_vector)
        bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
        return maxlen, wordtoix, ixtoword, bias_init_vector
