"""
Map the image file name and image index to the captions (sorted glob).
The 'image_index' key is super important because the feature maps' 
 index is corresponding to the file name

For example:
    feature.npy shape = (82783, 14, 14, 512)
    the first row of feature.npy is the first image's (sorted glob) feature map
"""
import pdb
import json
import glob
import cPickle

from tqdm import tqdm

COCO_TRAIN_ANNOTATIONS_JSON_PATH = '/home/markd/data/mscoco/annotations/captions_train2014.json'
COCO_TEST_FOLDER_PATH = '/home/markd/data/mscoco/testLSML_20548/'

"""Deal with training set now"""
with open(COCO_TRAIN_ANNOTATIONS_JSON_PATH, 'r') as f:
    info = json.load(f)

imgs = info[u'images']
annos = info[u'annotations']
imgs = sorted(imgs, key=lambda k: k[u'id'])
annos = sorted(annos, key=lambda k: k[u'image_id'])

# add the index key and remove all other redundant keys
for i, im in enumerate(imgs):
    del im[u'license']
    del im[u'coco_url']
    del im[u'height']
    del im[u'width']
    del im[u'date_captured']
    del im[u'flickr_url']
    im['index'] = i
# remove redundant key
for a in annos:
    del a[u'id']

annos_ult = []
# my super efficient loop, because both are sorted so we can pop the first item!
for im in tqdm(imgs):
    while(len(annos) != 0):
        if im[u'id'] == annos[0][u'image_id']:
            annos_ult.append({'caption':        annos[0][u'caption'].encode('utf8'),\
                              'image_name':     im[u'file_name'].encode('utf8'),\
                              'image_id':       im[u'id'],\
                              'image_index':    im['index']})
            annos.pop(0)
        else:
            break
cPickle.dump(annos_ult, open('train_82783_order.pkl', 'wb')) 

"""By far we are done with training set
We still need this to map the corresponding features even though
 we don't have to deal with captions.
Remember to deal with '/path/to/data/12345.jpg' to split it to 
'12345.jpg'
"""
test_files = sorted(glob.glob(COCO_TEST_FOLDER_PATH+'*'))
pure_files = []
for t in test_files:
    pure_files.append(t.split('/')[-1])
cPickle.dump(pure_files, open('test_20548_order.pkl', 'wb'))
