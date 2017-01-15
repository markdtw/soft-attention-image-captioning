from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

from tqdm import tqdm
import tensorflow as tf
import numpy as np

from utils import Data_loader 
from configs import config_test
from model import SoftAttentionModel 
from vgg import vgg19
from vgg import vgg_utils

def eval_20548(data_loader, context, sentence, sess):
    out_csv_f = open('generated.csv', 'w')
    out_csv_f.write('img_id,caption\n')
    for f in tqdm(data_loader.test_csv):
        conv5_4 = data_loader.imgs_vgg[data_loader.test_20548_order.index(f)]
        conv5_4 = conv5_4.reshape(-1, params.ctx_shape[0], params.ctx_shape[1])

        captions_ix = sess.run(sentence, feed_dict={context:conv5_4})
        captions_wd = [data_loader.index_to_word[x[0]] for x in captions_ix]
        try:
            captions_wd = ' '.join(captions_wd[:captions_wd.index('.')])
        except:
            captions_wd = ' '.join(captions_wd)
            #captions_wd = 'HOLY SHIT SOMETHING IS WRONG'

        out_csv_f.write(f + ',' + captions_wd + '\n')

    out_csv_f.close()

def test(params, data_loader):
    g = tf.Graph()
    params.bias_init = False
    model = SoftAttentionModel(params, data_loader.n_words, data_loader.maxlen)
    context, sentence = model.gen_caption()
    print ('Using model: {}'.format(params.model_path))
    # GO GO GO
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, params.model_path)
    if params.eval_all:
        eval_20548(data_loader, context, sentence, sess)
    else:
        img = vgg_utils.load_image(params.img_path)
        batch = img.reshape((1, 224, 224, 3))
        img_ph = tf.placeholder(tf.float32, [1, 224, 224, 3])
        vgg_model = vgg19.Vgg19()
        vgg_model.build(img_ph)
        conv5_4 = sess.run(vgg_model.conv5_4, feed_dict={img_ph: batch})
        conv5_4 = conv5_4.reshape(-1, params.ctx_shape[0], params.ctx_shape[1])

        captions_ix = sess.run(sentence, feed_dict={context: conv5_4})
        captions_wd = [data_loader.index_to_word[x[0]] for x in captions_ix]
        try:
            captions_wd = ' '.join(captions_wd[:captions_wd.index('.')])
        except ValueError:
            captions_wd = ' '.join(captions_wd)
        print (captions_wd)

if __name__ == '__main__':
    params = config_test()
    data_loader = Data_loader(params, True)
    print ('Data prepared, going to test...')
    test(params, data_loader)
