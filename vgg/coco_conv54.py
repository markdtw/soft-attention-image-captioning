"""
*Requires super large CPU memory*

This program extracts features from vgg conv5-4 layer, shape is (14, 14, 512)
and save it to a npy file.
"""
import glob
import pdb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' # turn off tensorflow GPU logging

import numpy as np
import tensorflow as tf
import skimage.color
from tqdm import tqdm

import vgg19
import utils

# len all = 82783 /199 = 416
all_train = sorted(glob.glob('train2014/*')) # mscoco images folder
n_all = len(all_train)

def get_batchimgs_loop(offset, batch_size):
    batch = []
    for index in xrange(batch_size):
        raw = utils.load_image(all_train[index+offset])
        if raw.shape == (224, 224):
            raw = skimage.color.gray2rgb(raw)
        batch.append(raw)
    return batch

def go():
    batch_size = 199
    all_conv5_4 = np.zeros((0, 14, 14, 512))
    with tf.Graph().as_default():
        with tf.Session() as sess:
            images = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
            vgg = vgg19.Vgg19()
            vgg.build(images)
            for index in tqdm(xrange(0, n_all, batch_size)):
                if index == n_all - batch_size + 1:
                    batch = get_batchimgs_loop(index, batch_size-1)
                    batch.append(np.zeros((1, 14, 14, 512)))
                else:
                    batch = get_batchimgs_loop(index, batch_size)
                feed_dict = {images: batch}
                conv5_4 = sess.run(vgg.conv5_4, feed_dict=feed_dict)
                all_conv5_4 = np.concatenate((all_conv5_4, conv5_4), axis=0)
            np.save('train_vggc54npf16.npy', all_conv5_4.astype(np.float16, copy=False))

go()
