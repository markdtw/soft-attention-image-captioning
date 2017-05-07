import re
import os
import pdb
import json
import time
import glob
import cPickle

import skimage.io
import skimage.color
import skimage.transform
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from cnns import inception_v4, vgg

slim = tf.contrib.slim

def load_and_process(path, cnn):
    img = skimage.io.imread(path)
    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    if cnn == 'inception':
        resized_img = skimage.transform.resize(crop_img, (299, 299), mode='reflect')
        resized_img -= 0.5
        resized_img *= 2.0
    elif cnn == 'vgg':
        _vgg_mean = [123.68, 116.78, 103.94]
        # resize to 224, 224
        resized_img = skimage.transform.resize(crop_img, (224, 224), mode='reflect')
        resized_img[:, :, 0] -= _vgg_mean[0]
        resized_img[:, :, 1] -= _vgg_mean[1]
        resized_img[:, :, 2] -= _vgg_mean[2]
    return resized_img

def extract_single(sess, img_path, cnn='inception'):
    """Extract features from a single image"""
    loaded_img = load_and_process(img_path, cnn)
    if cnn == 'inception':
        processed_images = tf.placeholder(tf.float32, [None, 299, 299, 3])
        with slim.arg_scope(inception_v4.arg_scope()):
            logits, _ = inception_v4.inception_v4(processed_images, is_training=False)
        init_fn = slim.assign_from_checkpoint_fn(('./cnns/inception_v4_imagenet.ckpt'),
            slim.get_model_variables('InceptionV4'))
    elif cnn == 'vgg':
        processed_images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        with slim.arg_scope(vgg.vgg_arg_scope()):
            logits, _ = vgg.vgg_19(processed_images, is_training=False)
        init_fn = slim.assign_from_checkpoint_fn(('./cnns/vgg_19_imagenet.ckpt'),
            slim.get_model_variables('vgg_19'))

    init_fn(sess)
    feat = sess.run(logits, feed_dict={processed_images: loaded_img[np.newaxis, :, :, :]})
    return feat

def extract_all(train_val='train', cnn='inception'):
    """Extract train/val features from vgg_19 or inception
    save mscoco train and val images to /home/mscoco/train2014 and /home/mscoco/val2014

    will dump a large npy file, prepare your memory
    """
    data2014 = sorted(glob.glob('/home/mscoco/'+train_val+'2014/*'))
    n_examples = len(data2014)
    batch_size = 64

    if cnn == 'inception':
        processed_images = tf.placeholder(tf.float32, [None, 299, 299, 3])
        with slim.arg_scope(inception_v4.arg_scope()):
            logits, _ = inception_v4.inception_v4(processed_images, is_training=False)
        init_fn = slim.assign_from_checkpoint_fn(('./cnns/inception_v4_imagenet.ckpt'),
            slim.get_model_variables('InceptionV4'))
    elif cnn == 'vgg':
        processed_images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        with slim.arg_scope(vgg.vgg_arg_scope()):
            logits, _ = vgg.vgg_19(processed_images, is_training=False)
        init_fn = slim.assign_from_checkpoint_fn(('./cnns/vgg_19_imagenet.ckpt'),
            slim.get_model_variables('vgg_19'))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init_fn(sess)

    feat_concat = []
    for i in tqdm(xrange(0, n_examples, batch_size)):
        batch = []
        for j in xrange(batch_size):
            if i+j < n_examples:
                batch.append(load_and_process(data2014[i+j], cnn))

        #pdb.set_trace()
        feat = sess.run(logits, feed_dict={processed_images: batch})
        feat = feat.astype(np.float16, copy=False)
        feat_concat.append(feat)

    np.save('data/'+train_val+'2014_'+cnn+'.npy', np.concatenate(feat_concat))


def finalCaptions():
    """Convert word to index

    caption = ['i', 'am', 'groot']
    new_cap = [9, 3, 53, 0, 0, 0, ..., 0] (length is equal to maxlen)
    mask    = [1, 1,  1, 1, 0, 0, ..., 0] (for computing loss)
    
    need data/coco_processed.json generated from processCaptions()
    need data/coco_dictionary.pkl generated from preProBuildWordVocab()
    """
    coco_processed = json.load(open('data/coco_processed.json', 'r'))
    coco_dictionary = cPickle.load(open('data/coco_dictionary.pkl', 'rb'))

    out = []
    for c in coco_processed:
        word_list = c['caption'].split(' ')
        mask_list = [1] * (len(word_list)+1) + [0] * (coco_dictionary['maxlen']-len(word_list))
        # done generating mask
        indx_list = []
        for word in word_list:
            if word in coco_dictionary['wordtoix']:
                indx_list.append(coco_dictionary['wordtoix'][word])
            else:
                indx_list.append(coco_dictionary['wordtoix']['#RARE#'])
        indx_list += [0] * (coco_dictionary['maxlen'] - len(indx_list) + 1) # pad 0 to maxlen
        new_c = {'caption': indx_list,\
                 'mask': mask_list,\
                 'file_path': c['file_path'],\
                 'image_id': c['image_id'],\
                 'image_ix': c['image_ix']}
        out.append(new_c)

    json.dump(out, open('data/coco_final.json', 'w')) 

def preProBuildWordVocab():
    """Borrowed from karpathy/neuraltalk again.
    Return:
        maxlen: longest caption length
        wordtoix: encode word to index
        ixtoword: decode from index to word
        bias_init_vector: for reducing perplexity
    
    need data/coco_processed.json generated from processCaptions()
    """
    coco_processed = json.load(open('data/coco_processed.json', 'r'))
    sentence_iterator = []
    for c in coco_processed:
        sentence_iterator.append(c['caption'])
    word_count_threshold = 5
    # count up all word counts so that we can threshold
    # this shouldnt be too expensive of an operation
    print ('Preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, ))
    t0 = time.time()
    word_counts = {}
    nsents = 0
    maxlen = 0
    for sent in sentence_iterator:
        nsents += 1
        sent_list = sent.split(' ')
        if len(sent_list) > maxlen:
            maxlen = len(sent_list)
        for w in sent_list:
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print ('Filtered words from %d to %d in %.2fs' % (len(word_counts), len(vocab), time.time() - t0))

    # with K distinct words:
    # - there are K+1 possible inputs (START token and all the words)
    # - there are K+1 possible outputs (END token and all the words)
    # we use ixtoword to take predicted indeces and map them to words for output visualization
    # we use wordtoix to take raw words and get their index in word vector matrix
    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    ixtoword[1] = '#RARE#'
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    wordtoix['#RARE#'] = 1
    ix = 2
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1
    
    # compute bias vector, which is related to the log probability of the distribution
    # of the labels (words) and how often they occur. We will use this vector to initialize
    # the decoder weights, so that the loss function doesnt show a huge increase in performance
    # very quickly (which is just the network learning this anyway, for the most part). This makes
    # the visualizations of the cost function nicer because it doesn't look like a hockey stick.
    # for example on Flickr8K, doing this brings down initial perplexity from ~2500 to ~170.
    word_counts['.'] = nsents
    word_counts['#RARE#'] = word_count_threshold
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    coco_dictionary = {'maxlen': maxlen,
                       'wordtoix': wordtoix,
                       'ixtoword': ixtoword,
                       'bias_init_vector': bias_init_vector}
    cPickle.dump(coco_dictionary, open('data/coco_dictionary.pkl', 'wb'))

def processCaptions():
    """Process the captions:
    1. remove all the non-alphanumerical characters
    2. remove multiple white spaces
    3. remove leading or trailing white spaces
    4. convert to lower case
    5. pair captions with their image informations
    6. sort by file_path (will be done first to have image_ix information)
    
    need data/coco_raw.json generated from concatCaptions()
    """
    coco_raw = json.load(open('data/coco_raw.json', 'r'))
    coco_raw = sorted(coco_raw, key=lambda k: k['file_path'])
    out = []
    image_ix = 0
    for raw in coco_raw:
        for c in raw['captions']:
            cap = re.sub(r'[^\w^\s]', '', c)
            cap = re.sub(r'\s+', ' ', cap)
            cap = cap.strip().lower()
            pair = {'caption': cap,\
                    'file_path': raw['file_path'],\
                    'image_id': raw['id'],\
                    'image_ix': image_ix}
            out.append(pair)
        image_ix += 1

    json.dump(out, open('data/coco_processed.json', 'w')) 

def concatCaptions():
    """Borrowed from karpathy/neuraltalk2/coco_preprocess.ipynb
    First concatenate train and val sets, then put all the captions of the same image together"""

    val = json.load(open('/home/mscoco/annotations/captions_val2014.json', 'r'))
    train = json.load(open('/home/mscoco/annotations/captions_train2014.json', 'r'))

    # combine all images and annotations together
    imgs = val['images'] + train['images']
    annots = val['annotations'] + train['annotations']

    # for efficiency lets group annotations by image
    itoa = {}
    for a in annots:
	imgid = a['image_id']
	if not imgid in itoa: itoa[imgid] = []
	itoa[imgid].append(a)

    # create the json blob
    out = []
    for i,img in enumerate(imgs):
	imgid = img['id']
	
	# coco specific here, they store train/val images separately
	loc = 'train2014' if 'train' in img['file_name'] else 'val2014'
	
	jimg = {}
	jimg['file_path'] = os.path.join(loc, img['file_name'])
	jimg['id'] = imgid
	
	sents = []
	annotsi = itoa[imgid]
	for a in annotsi:
	    sents.append(a['caption'])
	jimg['captions'] = sents
	out.append(jimg)
	
    json.dump(out, open('data/coco_raw.json', 'w'))

if __name__ == '__main__':
    """Run the following functions in this order to get coco_final.json for training"""
    #concatCaptions()
    #processCaptions()
    #preProBuildWordVocab()
    #finalCaptions()
    """Run this to extract feature"""
    extract_all(train_val='train', cnn='vgg')
