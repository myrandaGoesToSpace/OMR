# OMR Model 
# Goal: recognize images of music excerpts

# Modules
import torch
from torch.autograd import Variable
import numpy as np
import pylab as pl
import torch.nn.init as init
import torch.optim as optim
import torch.nn as nn

class cnn_model(torch.nn.Module):
    def __init__(self, batch_size, n_inputs, n_hidden, n_outputs):
        super(cnn_model, self).__init__()

        kernel_size = [3,3]

        self.conv1 = nn.Conv2d(32, 64, kernel_size = kernel_size)
        self.batch1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,128, kernel_size = kernel_size)
        self.batch2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,256, kernel_size = kernel_size)
        self.batch3 = nn.BatchNorm2d(256)

        self.act = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(2,2)


        seq_len = 1
        batch_size = 16
        input_size = 256 * 256 * 1



    def forward(self, x):

        # FORWARD PASS
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = self.act(x)
        x = self.pool(x)

        output = x

        return x

class rnn_model(nn.Module):
    def __init__(self, embed_size, hidden_size=512, vocab_size):
        super(rnn_model, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.rnn = nn.RNN(input_size = embed_size, hidden_size = hidden_size,num_layers = 2)
        self.fc = nn.Linear(hidden_size, vocab_size + 1)

    def forward(x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.rnn(x, (h0, c0))

        out = out.reshape(out.shape,[0], -1)
        out = self.fc(out)

        return out

class BasicRNN(nn.Module):
    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs):
        super(ImageRNN, self).__init__()
        
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        self.basic_rnn = nn.RNN(self.n_inputs, self.n_neurons) 
        
        self.FC = nn.Linear(self.n_neurons, self.n_outputs)
        
    def init_hidden(self,):
        # (num_layers, batch_size, n_neurons)
        return (torch.zeros(1, self.batch_size, self.n_neurons))
        
    def forward(self, X):
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        X = X.permute(1, 0, 2) 
        
        self.batch_size = X.size(1)
        self.hidden = self.init_hidden()
        
        # lstm_out => n_steps, batch_size, n_neurons (hidden states for each time step)
        # self.hidden => 1, batch_size, n_neurons (final state from each lstm_out)
        lstm_out, self.hidden = self.basic_rnn(X, self.hidden)      
        out = self.FC(self.hidden)
        
        return out.view(-1, self.n_outputs) # batch_size X n_output
'''
# TESTING PURPOSES ONLY 

def default_model_params(img_height, vocabulary_size):
    params = dict()
    params['img_height'] = img_height
    params['img_width'] = None
    params['batch_size'] = 16
    params['img_channels'] = 1
    params['conv_blocks'] = 4
    params['conv_filter_n'] = [32, 64, 128, 256]
    params['conv_filter_size'] = [ [3,3], [3,3], [3,3], [3,3] ]
    params['conv_pooling_size'] = [ [2,2], [2,2], [2,2], [2,2] ]
    params['rnn_units'] = 512
    params['rnn_layers'] = 2
    params['vocabulary_size'] = vocabulary_size
    return params

def model(params):

    data = np.array(shape = (None, params['img_height'], params['img_width'], params['img_channels'])

    width_reduction = 1
    height_reduction = 1

    x = data

    for i in range(params['conv_blocks']):

        x = torch.nn.Conv2d(params['conv_filter_in'][i][0], params['conv_filter_n'][i][1], kernel_size = params['conv_filter_size'])

        # ADD OTHER LAYERS

        width_reduction = width_reduction * params['conv_pooling_size'][i][1]
        height_reduction = height_reduction * params['conv_pooling_size'][i][0]

    

    # ADD RNN LAYERS


    return x
'''    
