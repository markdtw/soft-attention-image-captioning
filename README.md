# Soft Attention Image Captioning
Tensorflow implementation of [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) presented in ICML'15.

Huge re-factor from last update, compatible with tensorflow >= r1.0


## Prerequisites
- Python 2.7+
- [NumPy](http://www.numpy.org/)
- [Tensorflow r1.0+](https://www.tensorflow.org/install/)
- [Scikit-image](http://scikit-image.org/)
- [tqdm](https://pypi.python.org/pypi/tqdm)


## Data
- Training: [Microsoft COCO: Common Objects in Context](http://mscoco.org/dataset/#download) training and validation set


## Preparation
1. Clone this repo, create `data/` and `log/` folders:
```bash
git clone https://github.com/markdtw/soft-attention-image-captioning.git
cd soft-attention-image-captioning
mkdir data
mkdir log
```
2. Download and extract pre-trained `Inception V4` and `VGG 19` [from tf.slim](https://github.com/tensorflow/models/tree/master/slim) for feature extraction.  
   Save the ckpt files in `cnns/` as `inception_v4_imagenet.ckpt` and `vgg_19_imagenet.ckpt`.

3. We need the following files in our `data/` folder:

  - `coco_raw.json`
  - `coco_processed.json`
  - `coco_dictionary.pkl`
  - `coco_final.json`
  - `train2014_vgg(inception).npy` and `val2014_vgg(inception).npy`

   These files can be generated through `utils.py`, please refer to it before executing.


## Train
Train from scratch with default settings:
```bash
python main.py --train
```
Train from a pre-trained model from epoch X:
```bash
python main.py --train --model_path=log/model.ckpt-X
```
Check out tunable arguments:
```bash
python main.py
```

## Generate a caption
Using default(latest) model:
```bash
python main.py --generate --img_path=/path/to/image.jpg
```
Using model from epoch X:
```bash
python main.py --generate --img_path=/path/to/image.jpg --model_path=log/model.ckpt-X
```

## Result
Training...


## Others
- Features extracted are around 16 + 7.6 (train+val) GB. Make sure you have enough CPU memory when loading the data.
- GPU memory usage for batch_size 128 is around 8GB.
- Utilize `tf.while_loop` for rnn implementation, `tf.slim` for feature extraction from their [github page](https://github.com/tensorflow/models/tree/master/slim).
- GRU cell is implemented, use it by setting `--use_gru=True` when training. (not yet test though)
- Features can be extracted through [inceptionV4](https://arxiv.org/abs/1602.07261), if so, model.ctx_dim in `model.py` needs to be set to (64, 1536). (not yet test as well)
- Issues are welcome!


## Resources
- [Show, attend and tell slides](http://www.slideshare.net/eunjileee/show-attend-and-tell-neural-image-caption-generation-with-visual-attention)
- [Attention Mechanism Blog Post](https://blog.heuritech.com/2016/01/20/attention-mechanism/)

