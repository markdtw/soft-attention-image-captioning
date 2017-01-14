import os
import pdb
import math

import tensorflow as tf
import numpy as np

class SoftAttentionModel():
    
    def __init__(self, params, n_words, maxlen, bivector=None):
        self.params = params
        self.n_words = n_words
        self.steps = maxlen + 1
        # tf variables are initialized with the calling(accessing) priority for better understanding.
        # image attention weight is for the input feature map (14, 14, 512)
        # hidden attention weight and the previous attention bias deals with the encoded context
        self.image_att_W = self.init_weight(params.dim_ctx, params.dim_ctx, name='image_att_W')
        self.hidden_att_W = self.init_weight(params.dim_hid, params.dim_ctx, name='hidden_att_W')
        self.previous_att_b = self.init_bias(params.dim_ctx, name='previous_att_b')
        # the main attention weight and bias
        self.att_W = self.init_weight(params.dim_ctx, 1, name='att_W')
        self.att_b = self.init_bias(1, name='att_b')
        # the word embedding weights
        with tf.device('/cpu:0'):
            self.word_emb = tf.Variable(tf.random_uniform([n_words, params.dim_emb], -0.1, 0.1), name='word_emb')
        # the main LSTM weights
        self.lstm_W = self.init_weight(params.dim_emb, params.dim_hid*4, name='lstm_W')
        self.lstm_U = self.init_weight(params.dim_hid, params.dim_hid*4, name='lstm_U')
        self.lstm_b = self.init_bias(params.dim_hid*4, name='lstm_b')
        # the context to lstm weight matrix
        self.context_encoded_W = self.init_weight(params.dim_ctx, params.dim_hid*4, name='context_encoded_W')
        # the decode lstm weights
        self.decode_lstm_W = self.init_weight(params.dim_hid, params.dim_emb, name='decode_lstm_W')
        self.decode_lstm_b = self.init_bias(params.dim_emb, name='decode_lstm_b')
        self.decode_lstm_ctx_W = self.init_weight(params.dim_ctx, params.dim_emb, name='decode_lstm_ctx_W')
        self.decode_lstm_ctx_b = self.init_bias(params.dim_emb, name='decode_lstm_ctx_b')
        self.decode_lstm_word_W = self.init_weight(params.dim_emb, params.dim_emb, name='decode_lstm_word_W')
        self.decode_lstm_word_b = self.init_bias(params.dim_emb, name='decode_lstm_word_b')
        # the decode word weights
        self.decode_word_W = self.init_weight(params.dim_emb, n_words, name='decode_word_W')
        if params.bias_init:
            self.decode_word_b = tf.Variable(bivector.astype(np.float32), name='decode_word_b')
        else:
            self.decode_word_b = self.init_bias(n_words, name='decode_word_b')

    def build(self):
        """Build the model base on the paper: (https://arxiv.org/pdf/1502.03044v3.pdf)
        Show, Attend and Tell: Neural Image Caption Generation with Visual Attention, by Xu et al, ICML'15
        """
        # context is the visual context input, extracted from vgg's conv5-4, shape is 14*14*512 (196, 512).
        # sentence is the transformed-from-word-to-index caption we are going to predict.
        # mask is the corresponding sentence mask with its length, with 1 as activation and 0 otherwise.
        context = tf.placeholder(tf.float32, [self.params.batch_size, self.params.ctx_shape[0], self.params.ctx_shape[1]])
        sentence = tf.placeholder(tf.int32, [self.params.batch_size, self.steps])
        mask = tf.placeholder(tf.float32, [self.params.batch_size, self.steps])

        # context have to be flattened to do tf.matmul
        context_flatten = tf.reshape(context, [-1, self.params.dim_ctx])
        context_encoded = tf.matmul(context_flatten, self.image_att_W)
        context_encoded = tf.reshape(context_encoded, [-1, self.params.ctx_shape[0], self.params.ctx_shape[1]])

        # In the paper, the initial memory state (c) and hidden state (h) of the LSTM 
        #  are predicted by an average of the annotation vectors fed through two separate MLPs.
        c, h = self.init_lstm(tf.reduce_mean(context, 1))

        total_loss = 0.0
        for step in xrange(self.steps):
            # calculate the context for alpha, then we are pretty much done with the attention part
            # potential bug: context_encoded += tf.expand_dims(tf.matmul(h, self.hidden_att_W), 1) + self.previous_att_b
            context_encoded += tf.expand_dims(tf.matmul(h, self.hidden_att_W), 1) + self.previous_att_b
            context_encoded = tf.nn.tanh(context_encoded)
            context_encoded_flat = tf.reshape(context_encoded, [-1, self.params.dim_ctx])
            alpha = tf.matmul(context_encoded_flat, self.att_W) + self.att_b
            alpha = tf.reshape(alpha, [-1, self.params.ctx_shape[0]])
            alpha = tf.nn.softmax(alpha)

            weighted_context = tf.reduce_sum(context * tf.expand_dims(alpha, 2), 1)
            # next, we build the standard LSTM cells with the 
            # initialize word embedding matrix
            if step == 0:
                word_emb = tf.zeros([self.params.batch_size, self.params.dim_emb])
            else:
                tf.get_variable_scope().reuse_variables()
                with tf.device('/cpu:0'):
                    word_emb = tf.nn.embedding_lookup(self.word_emb, sentence[:, step-1])
            x_t = tf.matmul(word_emb, self.lstm_W) + self.lstm_b
            lstm_pack = tf.matmul(h, self.lstm_U) + x_t + tf.matmul(weighted_context, self.context_encoded_W)
            # LSTM standard equations, detail:
            #  http://cs231n.stanford.edu/slides/winter1516_lecture10.pdf (page.70)
            i, f, o, new_c = tf.split(1, 4, lstm_pack)
            i = tf.nn.sigmoid(i)
            f = tf.nn.sigmoid(f)
            o = tf.nn.sigmoid(o)
            new_c = tf.nn.tanh(new_c)
            c = i * new_c + f * c
            h = o * tf.nn.tanh(c)           # bugs in original code
            # calculate logits              # bugs in original code as well
            logits = tf.matmul(h, self.decode_lstm_W) + self.decode_lstm_b +\
                     tf.matmul(weighted_context, self.decode_lstm_ctx_W) + self.decode_lstm_ctx_b +\
                     tf.matmul(word_emb, self.decode_lstm_word_W) + self.decode_lstm_word_b
            logits = tf.nn.dropout(tf.nn.relu(logits), 0.5)
            logits_word = tf.matmul(logits, self.decode_word_W) + self.decode_word_b

            # convert the sentence to single word and then to one-hot vectors
            labels = tf.expand_dims(sentence[:, step], 1)
            indices = tf.expand_dims(tf.range(0, self.params.batch_size, 1), 1)
            concated = tf.concat(1, [indices, labels])
            onehot_labels = tf.sparse_to_dense(concated, tf.pack([self.params.batch_size, self.n_words]), 1.0, 0.0)

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits_word, onehot_labels)
            cross_entropy *= mask[:,step]

            current_loss = tf.reduce_sum(cross_entropy)
            total_loss += current_loss

        total_loss /= tf.reduce_sum(mask)
        self.var_summaries([weighted_context, total_loss])
        return total_loss, context, sentence, mask

    def gen_caption(self):
        context = tf.placeholder(tf.float32, [1, self.params.ctx_shape[0], self.params.ctx_shape[1]])
        c, h = self.init_lstm(tf.reduce_mean(context, 1))

        context_encoded = tf.matmul(tf.squeeze(context), self.image_att_W)
        word_emb = tf.zeros([1, self.params.dim_emb])

        sentence = []
        alpha_list = []
        logits_list = []
        for step in xrange(self.steps-1):
            context_encoded += tf.matmul(h, self.hidden_att_W) + self.previous_att_b
            context_encoded = tf.nn.tanh(context_encoded)

            alpha = tf.matmul(context_encoded, self.att_W) + self.att_b
            alpha = tf.reshape(alpha, [-1, self.params.ctx_shape[0]])
            alpha = tf.nn.softmax(alpha)
            alpha = tf.reshape(alpha, (self.params.ctx_shape[0], -1))
            weighted_context = tf.reduce_sum(tf.squeeze(context) * alpha, 0)
            weighted_context = tf.expand_dims(weighted_context, 0)
            
            alpha_list.append(alpha)

            x_t = tf.matmul(word_emb, self.lstm_W) + self.lstm_b
            lstm_pack = tf.matmul(h, self.lstm_U) + x_t + tf.matmul(weighted_context, self.context_encoded_W)
            i, f, o, new_c = tf.split(1, 4, lstm_pack)
            i = tf.nn.sigmoid(i)
            f = tf.nn.sigmoid(f)
            o = tf.nn.sigmoid(o)
            new_c = tf.nn.tanh(new_c)
            c = i * new_c + f * c
            h = o * tf.nn.tanh(c)           # bugs in original code

            logits = tf.matmul(h, self.decode_lstm_W) + self.decode_lstm_b +\
                     tf.matmul(weighted_context, self.decode_lstm_ctx_W) + self.decode_lstm_ctx_b +\
                     tf.matmul(word_emb, self.decode_lstm_word_W) + self.decode_lstm_word_b
            logits = tf.nn.relu(logits)
            logits_word = tf.matmul(logits, self.decode_word_W) + self.decode_word_b
            max_prob_word = tf.argmax(logits_word, 1)

            with tf.device('/cpu:0'):
                word_emb = tf.nn.embedding_lookup(self.word_emb, max_prob_word)

            sentence.append(max_prob_word)
            logits_list.append(logits_word)

        return context, sentence, logits_list, alpha_list

    def var_summaries(self, varlist):
        for var in varlist:
            tf.summary.scalar(var.name, var)
            tf.summary.histogram(var.name+'_hist', var)
    def init_lstm(self, mean_ctx):
        init_mW = self.init_weight(self.params.dim_ctx, self.params.dim_hid, name='init_mW')
        init_mb = self.init_bias(self.params.dim_hid, name='init_mb')
        init_hW = self.init_weight(self.params.dim_ctx, self.params.dim_hid, name='init_hW')
        init_hb = self.init_bias(self.params.dim_hid, name='init_hb')

        init_c = tf.nn.tanh(tf.matmul(mean_ctx, init_mW) + init_mb)
        init_h = tf.nn.tanh(tf.matmul(mean_ctx, init_hW) + init_hb)
        return init_c, init_h
    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)
    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)
