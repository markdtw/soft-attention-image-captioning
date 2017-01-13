import os
import pdb

from tensorflow.nn.rnn_cell import BasicLSTMCell
import tensorflow as tf
import numpy as np

class SoftAttentionModel():
    
    def __init__(self, params):
        self.params = params
        pass
    def build(self):
        """Build the model base on the paper: (https://arxiv.org/pdf/1502.03044v3.pdf)
        Show, Attend and Tell: Neural Image Caption Generation with Visual Attention, by Xu et al, ICML'15
        """
        # context is the visual context input, extracted from vgg's conv5-4, shape is 14*14*512 (196, 512).
        # sentence is the transformed-from-word-to-index caption we are going to predict.
        # mask is the corresponding sentence mask with its length, with 1 as activation and 0 otherwise.
        context = tf.placeholder(tf.float32, [self.batch_size, self.ctx_shape[0], self.ctx_shape[1]])
        sentence = tf.placeholder(tf.int32, [self.batch_size, self.maxlen+1])
        mask = tf.placeholder(tf.float32, [self.batch_size, self.maxlen+1])

        # In the paper, the initial memory state (c) and hidden state (h) of the LSTM 
        #  are predicted by an average of the annotation vectors fed through two separate MLPs.
        c0, h0 = self.LSTM_init(tf.reduce_mean(context, 1))

        context_flatten = tf.reshape(context, [-1, self.dim_ctx])
        context_encoded = tf.matmul(context_flatten, self.image_att_W)
        context_encoded = tf.reshape(context_encode, [-1, self.ctx_shape[0], ctx_shape[1]])

        loss = 0.0

        for index in xrange(self.maxlen+1):

