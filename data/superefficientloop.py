"""Map the image file name and index to the captions (all sorted)
The 'image_index' key in the final output file is super important
 because the feature maps' index is corresponding to the file name (sorted in order)

Please run in python 3 (The output file is still python2 compatible)
"""
import pdb
import json
import _pickle as cPickle

from tqdm import tqdm

with open('/home/markd/data/mscoco/annotations/captions_train2014.json', 'r') as f:
    info = json.load(f)

imgs = info['images']
annos = info['annotations']
imgs = sorted(imgs, key=lambda k: k['id'])
annos = sorted(annos, key=lambda k: k['image_id'])

# add the index key and remove all other redundant keys
for i, im in enumerate(imgs):
    del im['license']
    del im['coco_url']
    del im['height']
    del im['width']
    del im['date_captured']
    del im['flickr_url']
    im['index'] = i
# remove redundant key
for a in annos:
    del a['id']

annos_ult = []
# my super efficient loop, because it's sorted so we can pop the first item!
for im in tqdm(imgs):
    while(len(annos) != 0):
        if im['id'] == annos[0]['image_id']:
            annos_ult.append({'caption':        annos[0]['caption'],\
                              'image_id':       im['id'],\
                              'image_index':    im['index'],\
                              'image_name':     im['file_name']})
            annos.pop(0)
        else:
            break
# save to protocol 2 for python 2 compatibility
cPickle.dump(annos_ult, open('train_captions_ultimate.pkl', 'wb'), protocol=2) 
