from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

from six.moves import xrange
import tensorflow as tf
import numpy as np

from utils import Data_loader 
from configs import config_test
from model import SoftAttentionModel 
from vgg import vgg19
from vgg import vgg_utils

def test(params, data_loader):
    g = tf.Graph()
    params.bias_init = False
    model = SoftAttentionModel(params, data_loader.n_words, data_loader.maxlen)
    context, sentence, logits_list, alpha_list = model.gen_caption()
    print ('Using model: {}'.format(params.model_path))
    # GO GO GO
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, params.model_path)
    if not params.eval_all:
        img = vgg_utils.load_image(params.img_path)
        batch = img.reshape((1, 224, 224, 3))
        img_ph = tf.placeholder(tf.float32, [1, 224, 224, 3])
        vgg_model = vgg19.Vgg19()
        vgg_model.build(img_ph)
        conv5_4 = sess.run(vgg_model.conv5_4, feed_dict={img_ph: batch}).reshape(-1, params.ctx_shape[0], params.ctx_shape[1])

        captions_ix = sess.run(sentence, feed_dict={context: conv5_4})
        captions_wd = [data_loader.index_to_word[x[0]] for x in captions_ix]
        captions_wd = ' '.join(captions_wd[:captions_wd.index('.')])
        print (captions_wd)
    else:
        pass

if __name__ == '__main__':
    params = config_test()
    data_loader = Data_loader(params, True)
    print ('Data prepared, going to test...')
    test(params, data_loader)
