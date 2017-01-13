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
#import tensorflow as tf
import numpy as np

from utils import Data_loader 
from configs import config_train
from model import SoftAttentionModel 

def train(params, data_loader):
    model = SoftAttentionModel(params, data_loader.n_words, data_loader.maxlen, data_loader.bivector)
    loss, context, sentence, mask = model.build()
    train_op = tf.train.RMSPropOptimizer(params.learning_rate)
    merged = tf.summary.merge_all()

    init_op = tf.global_variables_initializer()
    saver = tf.trian.Saver()
    # Go go lamigo
    sess = tf.Session()
    sess.run(init_op)
    train_writer = tf.train.SummaryWriter(params.log_dir, sess.graph)
    if params.pretrained_path is not None:
        print ('Starting with pretrained model: {}'.format(params.pretrained_path))
        saver.restore(sess, params.pretrained_path)

if __name__ == '__main__':
    params = config_train()
    data_loader = Data_loader(params)
    pdb.set_trace()
    #train(params, data_loader)
