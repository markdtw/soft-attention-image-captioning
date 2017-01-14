from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import pdb
import sys 
import time
import gc

from six.moves import xrange
import tensorflow as tf
import numpy as np

from utils import Data_loader 
from configs import config_train
from model import SoftAttentionModel 

def train(params, data_loader):
    g = tf.Graph()
    model = SoftAttentionModel(params, data_loader.n_words, data_loader.maxlen, data_loader.bivec)
    loss, context, sentence, mask = model.build()
    train_op = tf.train.RMSPropOptimizer(params.learning_rate).minimize(loss)
    merged = tf.summary.merge_all()

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # GO GO GO
    sess = tf.Session()
    sess.run(init_op)
    train_writer = tf.summary.FileWriter(params.log_dir, g)
    if params.pretrained_path is not None:
        print ('Starting with pretrained model: {}'.format(params.pretrained_path))
        saver.restore(sess, params.pretrained_path)

    for epoch in xrange(params.epoch):
        for bat in xrange(data_loader.num_batches):
            context_batch, sequence_batch, masks_batch = data_loader.next_batch()
            feed_dict = {context: context_batch,
                         sentence: sequence_batch,
                         mask: masks_batch}
            _, loss_val = sess.run([train_op, loss], feed_dict=feed_dict)
            if bat % 10 == 0:
                print ('epoch: %03d, batch: %04d, loss: %.3f' % (epoch, bat, loss_val))

        saver.save(sess, params.log_dir+'model-epoch-{}'.format(epoch), global_step=epoch)

if __name__ == '__main__':
    params = config_train()
    data_loader = Data_loader(params)
    print ('Data prepared, building graphs...')
    train(params, data_loader)
