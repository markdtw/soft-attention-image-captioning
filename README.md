# Soft Attention Captioning

Tensorflow implementation of [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) presented in ICML'15.

This repository is highly based on [jazzsaxmafia/show_attend_and_tell.tensorflow](https://github.com/jazzsaxmafia/show_attend_and_tell.tensorflow) with few bugs fixed according to the paper. The bugs include LSTM and loss computation errors, though they are quite trivial.

## Prerequisites
- python 2.7+
- [Tensorflow 0.12](https://www.tensorflow.org/get_started/os_setup)
- [scikit-image](http://scikit-image.org/) (for feature extraction)
- tqdm
- pandas

## Data
- Training: [Microsoft COCO: Common Objects in Context](http://mscoco.org/dataset/#download) training set
- Testing: custom testing set based on `data/test.csv` (20548 images) crawled with MSCOCO API.

## Preprocessing
We need 3 things prepared before we train the model:

- The extracted image features from vgg19-conv5_4 of shape (14, 14, 512)
- Training captions with respect to the features
- Create an empty folder `log/` for tensorflow

Extracts the features by executing ```python vgg/coco_conv54.py```, which **requires heavy CPU memory usage (up to 80GB) and time (up to 5 hrs)**. The extracted features are too large that I decided to use np.float16 as the final type. The vgg model is from [machrisaa/tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg). To generate the needed captions for my implementation, please refer to `data/map_features.py`. This program generate two files: `train_82783_order.pkl`, `test_20548_order.pkl`.

After the preprocessing steps, we should have `train_82783_order.pkl`, `train_82783_vggc54npf16.npy` in the `data/` directory.

## Train
```python
python train.py
```
Tunable parameters in configs.py

## Test
Single image test:
```
python test.py --img_path=/path/to/image.jpg
```
Generate `generated.csv` for all the images in `data/test.csv`. This need you to extracts the features again...
```
python test.py --eval_all=True
```
Model can be designated by passing argument: ```python test.py --model_path=/path/to/model-epoch-n```. **Notice it is *model-epoch-n*, not *model-epoch-n.meta* nor *model-epoch-n.data* **.

## Evaluation
Sorry, didn't write.

## Resources
These helps me a lot when building the model besides the original paper:

- [Attention Mechanism Blog Post](https://blog.heuritech.com/2016/01/20/attention-mechanism/)
- [Show, attend and tell slides](http://www.slideshare.net/eunjileee/show-attend-and-tell-neural-image-caption-generation-with-visual-attention)
- [School Project Page](http://datalab-lsml.appspot.com/lectures/02-Image-Caption.html)

## Acknowledgments
Code based highly on [jazzsaxmafia/show_attend_and_tell.tensorflow](https://github.com/jazzsaxmafia/show_attend_and_tell.tensorflow)<br>
VGG model from [machrisaa/tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg)
