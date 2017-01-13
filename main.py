from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import pdb
import sys 
import time
import gc

from six.moves import xrange
#import tensorflow as tf
import numpy as np

from utils import Data_loader 
from configs import config_train
from model import SoftAttentionModel 

def train(params, data_loader):

    model = SoftAttentionModel(params)
    loss, context, sentence, mask = model.build()

if __name__ == '__main__':
    params = config_train()
    data_loader = Data_loader(params)
    train(params, data_loader)
