from __future__ import print_function

import sys
import os
import time
import random
import numpy as np
import theano
import theano.tensor as T
from theano.tensor import *
import lasagne
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
from lasagne.layers.shape import PadLayer

tensor5 = TensorType('float32', (False,)*5)

cfg = {
       'num_epochs' : 50,
       'learning_rate' : 0.001,
       'momentum' : 0.9,
       'input_T': tensor5, 
       'batch_size' : 5,
       'dims': (1,120,36,36),
       'randomSeed': 12345,
       'num_classes':6,
       }
       
def get_model(input_var=None):
    '''
    Builds C3D model
    Returns
    -------
    '''
    net = {} #
    net['input'] = lasagne.layers.InputLayer(shape=(None, )+cfg['dims'], input_var=input_var)
    # ----------- 1st layer group ---------------
    net['conv1'] = Conv3DDNNLayer(net['input'], 32, (5,5,5), 
                                  pad=1,nonlinearity=lasagne.nonlinearities.rectify,flip_filters=False)
    net['pool1']  = MaxPool3DDNNLayer(net['conv1'],pool_size=(2,2,2),stride=(2,2,2))
    net['drop1'] = lasagne.layers.dropout(net['pool1'], p=.3),
    # ------------- 2nd layer group --------------
    net['conv2'] = Conv3DDNNLayer(net['pool1'], 32, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
    net['pool2']  = MaxPool3DDNNLayer(net['conv2'],pool_size=(2,2,2),pad=(1,1,1),stride=(2,2,2))
    net['drop2'] = lasagne.layers.dropout(net['pool2'], p=.3),
    net['flat'] = lasagne.layers.FlattenLayer(net['drop2'],outdim=2)
    net['fc1']  = lasagne.layers.DenseLayer(net['flat'], num_units=32, nonlinearity=lasagne.nonlinearities.rectify)
    net['drop3'] = lasagne.layers.dropout(net['fc1'], p=.3),
    net['out']  = lasagne.layers.DenseLayer(net['drop3'], num_units=cfg['num_classes'], nonlinearity=lasagne.nonlinearities.softmax)
    
    print('====')
    s = lasagne.layers.get_output_shape(net['input'],(1,)+cfg['dims'])
    print(s)
    for n in ['conv1','pool1','conv2','pool2','flat']:#:,'fc1','out']:
       s = lasagne.layers.get_output_shape(net[n],s)
       print(n,s)    
    print('====')
    return net['out']
    
# https://github.com/Lasagne/Recipes/blob/master/modelzoo/c3d.py