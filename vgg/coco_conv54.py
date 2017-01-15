import glob
import pdb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import numpy as np
import tensorflow as tf
import skimage.color
from tqdm import tqdm

import vgg19
import utils

# len all_train = 82783 = 413*200 + 183
# len all_test = 20548 = 102*200 + 148
# len all_val = 40504 = 202*200 + 104
all_images = sorted(glob.glob('val2014/*'))
n_all = len(all_images)

def get_batchimgs_loop(offset, batch_size):
    batch = []
    for index in xrange(batch_size):
        print (index+offset)
        raw = utils.load_image(all_images[index+offset])
        if raw.shape == (224, 224):
            raw = skimage.color.gray2rgb(raw)
        batch.append(raw)
    return batch

def go():
    batch_size = 200
    all_conv5_4 = np.zeros((0, 14, 14, 512))
    with tf.Graph().as_default():
        with tf.Session() as sess:
            images = tf.placeholder(tf.float32, [None, 224, 224, 3])
            vgg = vgg19.Vgg19()
            vgg.build(images)
            for index in tqdm(xrange(0, n_all, batch_size)):
                if index == 40400:
                    batch = get_batchimgs_loop(index, 104)
                else:
                    batch = get_batchimgs_loop(index, batch_size)
                feed_dict = {images: batch}
                conv5_4 = sess.run(vgg.conv5_4, feed_dict=feed_dict)
                all_conv5_4 = np.concatenate((all_conv5_4, conv5_4), axis=0)
            np.save('valmylife.npy', all_conv5_4)

go()
