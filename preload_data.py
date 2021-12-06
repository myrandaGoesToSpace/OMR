# Preload data using cv2
# So that I can use NVIDIA GPU in PyTorch

import cv2
import numpy as np
import random
from primus import CTC_PriMuS
import pickle


def preload_data(corpus=None, set=None, vocabulary=None, params=None):
    
    if corpus == None:
        corpus = './Data/package'# PATH
        set = 'Data/train.txt' 
        vocabulary = 'Data/vocabulary_semantic.txt'  

    primus = CTC_PriMuS(corpus, set, vocabulary, semantic = True, val_split = 0.1)

    if params == None:
        # Default params
        params = dict()
        params['img_height'] = 128
        params['img_width'] = None
        params['batch_size'] = 16
        params['img_channels'] = 1
        params['conv_blocks'] = 4
        params['conv_filter_n'] = [32, 64, 128, 256]
        params['conv_filter_size'] = [ [3,3], [3,3], [3,3], [3,3] ]
        params['conv_pooling_size'] = [ [2,2], [2,2], [2,2], [2,2] ]
        params['rnn_units'] = 512
        params['rnn_layers'] = 2
        params['vocabulary_size'] = primus.vocabulary_size
        params['max_width'] = 1500

    len_train = len(primus.training_list)
    len_valid = len(primus.validation_list)

    with open('pickled_data.txt', 'wb') as fo:

        for i in range(0, len_train + len_valid):
            batch = primus.nextBatch(params)
            pickle.dump(batch, fo)

    return
