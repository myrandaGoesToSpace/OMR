# Train model

# Modules
import torch
from torch.autograd import Variable
import numpy as np
import pylab as pl
import torch.nn.init as init
import torch.optim as optim
import argparse
import os

# Local modules
import ctc_utils
import crnn_model
from primus import CTC_PriMuS

def train():

    # Use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load primus data
    primus = CTC_PriMuS(args.corpus, args.set, args.voc, args.semantic, val_split = 0.1)
    

    # Parameters
    img_height = 128
    max_epochs = 1000
    dropout = 0.5

    # Set model parameters and model
    params = crnn_model.default_model_params(img_height, primus.vocabulary_size)

    model = crnn_model.model(params) # ADD MODEL

    # criterion
    learning_rate = 0.001
    criterion = torch.nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate) ## ADD MODEL PARAMS
   

    for epoch in range(max_epochs):
        train_loss = 0.
        train_acc = 0.
        
        valid_loss = 0.
        valid_acc = 0.

        batch = primus.nextBatch(params)
        data = batch['inputs']
        targets = ctc_utils.sparse_tuple_from(batch['targets'])

        output =
    

    return

