import os
import pdb
import math

import tensorflow as tf
import numpy as np

from tensorflow.contrib.layers import xavier_initializer, l2_regularizer

class Model():

    def __init__(self, wemb_dim=256, hid_dim=512, seq_len=50, learning_rate=1e-3,
            batch_size=256, num_batches=0, num_words=0, biivector=None, use_gru=False, inference=False):
        self.wemb_dim = wemb_dim
        self.hid_dim = hid_dim
        self.seq_len = seq_len if not inference else seq_len - 1
        self.lr = learning_rate
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_words = num_words
        self.biivector = biivector
        self.use_gru = use_gru
        self.inference = inference

        self.lr_decay = 0.9
        self.l2_beta = 1e-2
        self.ctx_dim = (196, 512)
        
        self.g_step = tf.contrib.framework.get_or_create_global_step()
        if inference:
            self.batch_size = 1
        else:
            self.cap_ph = tf.placeholder(tf.int32, [self.batch_size, self.seq_len])
            self.mask_ph = tf.placeholder(tf.float32, [self.batch_size, self.seq_len])
        self.ctx_ph = tf.placeholder(tf.float32, [self.batch_size, self.ctx_dim[0], self.ctx_dim[1]])

        self._init_vars()

    def build(self):

        # maybe not necessary
        ai = tf.reshape(self.ctx_ph, [-1, self.ctx_dim[1]])             # shape = (batch_size*ctx_dim[0], ctx_dim[1])
        ai = tf.nn.xw_plus_b(ai, self.emb_att_w, self.emb_att_b)
        ai = tf.reshape(ai, [-1, self.ctx_dim[0], self.ctx_dim[1]])     # (bsize, ctx_dim[0], ctx_dim[1])
        #ai = self.ctx_ph

        # initialize hidden states using mean context
        mean_ctx = tf.reduce_mean(ai, 1)                        # (bsize, ctx_dim[1])
        init_c = tf.tanh(tf.matmul(mean_ctx, self.init_cW))     # (bsize, hid_dim)
        init_h = tf.tanh(tf.matmul(mean_ctx, self.init_hW))
        Ey_prev = tf.zeros([self.batch_size, self.wemb_dim])

        output_array = tf.TensorArray(dtype=tf.float32, size=self.seq_len)
        
        def _recurrent_body(i, ai, Ey_prev, c, h, output_array):
            # the attention model based on the alignment model by Bahdanau et al, ICLR'15
            ai_ = tf.reshape(ai, [-1, self.ctx_dim[1]])                     # (bsize*ctx_dim[0], ctx_dim[1])
            Wai_ = tf.matmul(ai_, self.att_W)                       
            Wai_ = tf.reshape(Wai_, [-1, self.ctx_dim[0], self.ctx_dim[1]]) # (bsize, ctx_dim[0], ctx_dim[1])
            Uh_ = tf.expand_dims(tf.matmul(h, self.att_U), 1)               # (bsize, 1, ctx_dim[1])
            Wai_Uh_ = tf.reshape(tf.add(Wai_, Uh_), [-1, self.ctx_dim[1]])  # (bsize*ctx_dim[0], ctx_dim[1])
            att_ = tf.nn.xw_plus_b(tf.tanh(Wai_Uh_), self.att_va, self.att_vb) # (bsize*ctx_dim[0], 1)

            alpha_ = tf.nn.softmax(tf.reshape(att_, [-1, self.ctx_dim[0]])) # (bsize, ctx_dim[0])

            zt = tf.reduce_sum(tf.multiply(ai, tf.expand_dims(alpha_, 2)), 1)    # (bsize, ctx_dim[1])

            if self.use_gru:
                h = self._GRUCell(Ey_prev, h, zt)
            else:
                c, h = self._LSTMCell(Ey_prev, h, zt, c)
            
            Lh = tf.matmul(h, self.rnn_oLh)     # (bsize, wemb_dim)
            Lz = tf.matmul(zt, self.rnn_oLz)    # (bsize, wemb_dim)
            output = tf.nn.xw_plus_b(tf.add_n([Ey_prev, Lh, Lz]), self.rnn_oLo, self.rnn_ob)   # (bsize, num_words)
            output_array = output_array.write(i, output)
            
            if self.inference:
                max_prob_word = tf.argmax(tf.nn.softmax(output), 1)
                Ey_prev = tf.nn.embedding_lookup(self.wemb, max_prob_word)
            else:
                Ey_prev = tf.nn.embedding_lookup(self.wemb, self.cap_ph[:, i])

            return i+1, ai, Ey_prev, c, h, output_array

        _, _, _, _, _, self.output_array = tf.while_loop(
            cond=lambda i, *_: i < self.seq_len,
            body=_recurrent_body,
            loop_vars=(tf.constant(0, tf.int32), ai, Ey_prev, init_c, init_h, output_array)
        )
        self.output_array = tf.transpose(self.output_array.stack(), [1, 0, 2])  # (bsize, seq_len, num_words)
        self.output_argmax = tf.squeeze(tf.argmax(tf.nn.log_softmax(self.output_array), 2)) # (seq_len)

    def loss(self):
        def _step_loss(i, total_xen_loss):
            cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(self.cap_ph[:, i], self.output_array[:, i, :])
            cross_entropy_loss *= self.mask_ph[:, i]
            return i+1, total_xen_loss + cross_entropy_loss

        _, cross_entropy_loss = tf.while_loop(
            cond=lambda i, *_: i < self.seq_len,
            body=_step_loss,
            loop_vars=(tf.constant(0, tf.int32), tf.constant(np.zeros(self.batch_size), tf.float32))
        )
        self.cross_entropy_loss_op = tf.reduce_sum(cross_entropy_loss) / tf.reduce_sum(self.mask_ph)
        self.reg_loss_op = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss_op = self.cross_entropy_loss_op + self.reg_loss_op

    def train(self):
        self.lr = tf.train.exponential_decay(self.lr, self.g_step,
                self.batch_size*self.num_batches, self.lr_decay, staircase=True)
        return self.optimizer(self.lr).minimize(self.loss_op, global_step=self.g_step)
    
    def optimizer(self, *args):
        return tf.train.AdamOptimizer(*args)

    def _LSTMCell(self, Ey_prev, h_prev, zt, c_prev):
        """
        Ey_prev  :  (bsize, wemb_dim)
        h_prev  :   (bsize, hid_dim)
        zt  :       (bsize, ctx_dim[1])
        c_prev  :   (bsize, hid_dim)
        """
        pack = tf.add_n([tf.matmul(Ey_prev, self.lstm_W), tf.matmul(h_prev, self.lstm_U), tf.matmul(zt, self.lstm_Z)])
        pack_with_bias = tf.nn.bias_add(pack, self.lstm_b)   # (bsize, hid_dim * 4)
        i, f, o, g = tf.split(pack_with_bias, num_or_size_splits=4, axis=1) # (bsize, hid_dim)
        i = tf.sigmoid(i)
        f = tf.sigmoid(f)
        o = tf.sigmoid(o)
        g = tf.tanh(g)
        c = tf.add(tf.multiply(f, c_prev), tf.multiply(i, g))
        h = tf.multiply(o, tf.tanh(c))
        return c, h

    def _GRUCell(self, Ey_prev, h_prev, zt):
        """
        Ey_prev  :  (bsize, wemb_dim)
        h_prev  :   (bsize, hid_dim)
        zt  :       (bsize, ctx_dim[1])
        """
        packz = tf.add_n([tf.matmul(Ey_prev, self.gru_Wz), tf.matmul(h_prev, self.gru_Uz), tf.matmul(zt, self.gru_Zz)])
        packr = tf.add_n([tf.matmul(Ey_prev, self.gru_Wr), tf.matmul(h_prev, self.gru_Ur), tf.matmul(zt, self.gru_Zr)])
        z = tf.sigmoid(tf.nn.bias_add(packz, self.gru_b))
        r = tf.sigmoid(tf.nn.bias_add(packr, self.gru_b))
        h_head = tf.add_n([tf.matmul(Ey_prev, self.gru_Ww),
                           tf.matmul(tf.multiply(r, h_prev), self.gru_Uw),
                           tf.matmul(zt, self.gru_Zw)])
        h = tf.tanh(h_head)
        h = tf.multiply((1 - z), h_prev) + tf.multiply(z, h)
        return h

    def _init_vars(self):
        def get_w(name, shape):
            return tf.get_variable(name, shape,
                    initializer=xavier_initializer(), regularizer=l2_regularizer(self.l2_beta))
        def get_b(name, shape):
            return tf.get_variable(name, shape, initializer=xavier_initializer())
        self.emb_att_w = get_w('emb_att_w', [self.ctx_dim[1], self.ctx_dim[1]])
        self.emb_att_b = get_b('emb_att_b', [self.ctx_dim[1]])

        self.init_cW = get_w('init_cW', [self.ctx_dim[1], self.hid_dim])
        self.init_hW = get_w('init_hW', [self.ctx_dim[1], self.hid_dim])

        self.att_W = get_w('att_W', [self.ctx_dim[1], self.ctx_dim[1]])
        self.att_U = get_w('att_U', [self.hid_dim, self.ctx_dim[1]])
        self.att_va = get_w('att_va', [self.ctx_dim[1], 1])
        self.att_vb = get_b('att_vb', [1])

        self.wemb = tf.get_variable('wemb', [self.num_words, self.wemb_dim], initializer=xavier_initializer())

        if self.use_gru:
            self.gru_Wz = get_w('gru_Wz', [self.wemb_dim, self.hid_dim])
            self.gru_Uz = get_w('gru_Uz', [self.hid_dim, self.hid_dim])
            self.gru_Zz = get_w('gru_Zz', [self.ctx_dim[1], self.hid_dim])
            self.gru_Wr = get_w('gru_Wr', [self.wemb_dim, self.hid_dim])
            self.gru_Ur = get_w('gru_Ur', [self.hid_dim, self.hid_dim])
            self.gru_Zr = get_w('gru_Zr', [self.ctx_dim[1], self.hid_dim])
            self.gru_Ww = get_w('gru_Ww', [self.wemb_dim, self.hid_dim])
            self.gru_Uw = get_w('gru_Uw', [self.hid_dim, self.hid_dim])
            self.gru_Zw = get_w('gru_Zw', [self.ctx_dim[1], self.hid_dim])
            self.gru_b = get_b('gru_b', [self.hid_dim])
        else:
            self.lstm_W = get_w('lstm_W', [self.wemb_dim, self.hid_dim * 4])
            self.lstm_U = get_w('lstm_U', [self.hid_dim, self.hid_dim * 4])
            self.lstm_Z = get_w('lstm_Z', [self.ctx_dim[1], self.hid_dim * 4])
            self.lstm_b = get_b('lstm_b', [self.hid_dim * 4])
        
        self.rnn_oLh = get_w('rnn_oLh', [self.hid_dim, self.wemb_dim])
        self.rnn_oLz = get_w('rnn_oLz', [self.ctx_dim[1], self.wemb_dim])
        self.rnn_oLo = get_w('rnn_oLo', [self.wemb_dim, self.num_words])

        if self.biivector is None:
            self.rnn_ob = get_b('rnn_ob', [self.num_words])
        else:
            self.rnn_ob = tf.get_variable('rnn_ob', [self.num_words], initializer=tf.constant_initializer(self.biivector))
