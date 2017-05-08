from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import sys
import argparse

import numpy as np
import tensorflow as tf

from model import Model
from utils import extract_single
from data_loader import Data_loader

def generate(args):

    data_loader = Data_loader(batch_size=1, bias_init=args.bias_init, train=False)

    model = Model(wemb_dim=args.wemb_dim, hid_dim=args.hid_dim, seq_len=data_loader.maxlen+1,
                  learning_rate=args.learning_rate, batch_size=1, num_batches=data_loader.num_batches,
                  num_words=data_loader.num_words, biivector=data_loader.biivector, use_gru=args.use_gru, inference=True)
    model.build()

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    if args.model_path is not None:
        print ('Using model: {}'.format(args.model_path))
        saver.restore(sess, args.model_path)
    else:
        latest_ckpt = tf.train.latest_checkpoint(args.logdir)
        print ('Did not provide model path, using latest: {}'.format(latest_ckpt))
        saver.restore(sess, latest_ckpt)

    feat = extract_single(sess, args.img_path, cnn='vgg')

    feed_dict = {model.ctx_ph: feat.reshape(-1, model.ctx_dim[0], model.ctx_dim[1])}
    captions_ix = sess.run(model.output_argmax, feed_dict=feed_dict)
    captions_wd = [data_loader.ixtoword[x] for x in captions_ix]
    try:
        captions_wd = ' '.join(captions_wd[:captions_wd.index('.')])
    except ValueError:
        captions_wd = ' '.join(captions_wd)
    print (captions_wd)
    print ('Sentence generated.')

def train(args):

    #if args.model_path is None:
    #    for f in [f for f in os.listdir(args.logdir)]: os.remove(os.path.join(args.logdir, f))

    data_loader = Data_loader(batch_size=args.batch_size, bias_init=args.bias_init)

    model = Model(wemb_dim=args.wemb_dim, hid_dim=args.hid_dim, seq_len=data_loader.maxlen+1,
                  learning_rate=args.learning_rate, batch_size=args.batch_size, num_batches=data_loader.num_batches,
                  num_words=data_loader.num_words, biivector=data_loader.biivector, use_gru=args.use_gru)
    model.build()
    model.loss()
    train_op = model.train()

    tf.summary.scalar('cross entropy loss', model.cross_entropy_loss_op)
    tf.summary.scalar('reg loss', model.reg_loss_op)
    tf.summary.scalar('loss', model.loss_op)
    merged_op = tf.summary.merge_all()
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    writer = tf.summary.FileWriter(args.logdir, sess.graph)
    sess.run(tf.global_variables_initializer())

    if args.model_path is not None:
        print ('Starting with pretrained model: {}'.format(args.model_path))
        saver.restore(sess, args.model_path)

    print ('Start training')
    for ep in xrange(args.epoch):
        for step in xrange(data_loader.num_batches):
            ctx_batch, cap_batch, mask_batch = data_loader.next_batch()
            feed_dict = {model.ctx_ph: ctx_batch.reshape(-1, model.ctx_dim[0], model.ctx_dim[1]),
                         model.cap_ph: cap_batch,
                         model.mask_ph: mask_batch}
            cross_entropy_loss, reg_loss, loss, _, summary = sess.run([
                    model.cross_entropy_loss_op, model.reg_loss_op, model.loss_op,
                    train_op, merged_op
            ], feed_dict=feed_dict) 
            
            writer.add_summary(summary, ep * data_loader.num_batches + step)
            if step % 100 == 0:
                print ('ep: %2d, step: %4d, xen_loss: %.4f, reg_loss: %.4f, loss: %.4f' %
                        (ep+1, step, cross_entropy_loss, reg_loss, loss))

        checkpoint_path = os.path.join(args.logdir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=ep+1)

    print ('Training done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='training the model.')
    parser.add_argument('--generate', action='store_true',
                        help='generating a caption from a given image (--img_path).')
    parser.add_argument('--learning_rate', metavar='', type=float, default=1e-3,        # learning_rate
                        help='initial learning rate.')
    parser.add_argument('--epoch', metavar='', type=int, default=30,                    # epoch
                        help='number of epochs.')
    parser.add_argument('--batch_size', metavar='', type=int, default=128,              # batch_size
                        help='batch size.')
    parser.add_argument('--wemb_dim', metavar='', type=int, default=256,                # wemb_dim
                        help='word embedding dimension.')
    parser.add_argument('--hid_dim', metavar='', type=int, default=256,                 # hid_dim
                        help='hidden layer dimension (RNN).')
    parser.add_argument('--use_gru', metavar='', type=bool, default=False,              # use_gru
                        help='gru cell (default LSTM cell).')
    parser.add_argument('--bias_init', metavar='', type=bool, default=True,             # bias_init
                        help='use bias init vector or not.')
    parser.add_argument('--logdir', metavar='', type=str, default='log',                # logdir
                        help='directory to save the trained models and summaries.')
    parser.add_argument('--model_path', metavar='', type=str, default=None,             # model_path
                        help='for pretraining or testing (if necessary).')
    parser.add_argument('--img_path', metavar='', type=str, default=None,               # img_path
                        help='if --generate is set, you have to provide the image path.')
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0: sys.exit('Unknown argument: {}'.format(unparsed))
    if args.train:
        train(args)
    if args.generate:
        generate(args)
    if not args.train and not args.generate:
        parser.print_help()
