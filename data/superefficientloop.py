"""
Map the image file name and image index to the captions (all sorted)
The 'image_index' key in the final output file is super important
 because the feature maps' index is corresponding to the file name (sorted in order)
"""
import pdb
import json
import cPickle

from tqdm import tqdm

with open('/home/markd/data/mscoco/annotations/captions_train2014.json', 'r') as f:
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
# my super efficient loop, because it's sorted so we can pop the first item!
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
cPickle.dump(annos_ult, open('2_train_captions_ultimate.pkl', 'wb')) 
