import numpy as np

import lasagne
import lasagne.layers

import voxnet

lr_schedule = { 0: 0.001,
                60000: 0.0001,
                400000: 0.00005,
                600000: 0.00001,
                }

cfg = {'batch_size' : 20,
       'learning_rate' : lr_schedule,
       'reg' : 0.001,
       'momentum' : 0.8,
       'dims' : (28,28),
       'n_channels' : 1,
       'n_classes' : 10,
       'batches_per_chunk': 10, 
       'max_epochs' : 40,
       'max_jitter_ij' : 3,
       'n_rotations' : 1,
       'checkpoint_every_nth' : 600,
       }

def get_model():
    dims, n_channels, n_classes = tuple(cfg['dims']), cfg['n_channels'], cfg['n_classes']
    shape = (None, n_channels)+dims

    l_in = lasagne.layers.InputLayer(shape=shape)
    l_conv1 = lasagne.layers.Conv2DLayer(
        incoming = l_in,
        num_filters = 32,
        filter_size = [5,5],
        stride = [1,1],
        W = lasagne.init.GlorotUniform(),
        nonlinearity = lasagne.nonlinearities.rectify,
        name =  'conv1'
        )
    l_pool1 = lasagne.layers.MaxPool2DLayer(
        incoming = l_conv1,
        pool_size = [2,2],
        name = 'pool1',
        )        
    l_drop1 = lasagne.layers.DropoutLayer(
        incoming = l_pool1,
        p = 0.5,
        name = 'drop1'
        )
    l_fc1 = lasagne.layers.DenseLayer(
        incoming = l_drop1,
        num_units = 128,
        W = lasagne.init.GlorotUniform(),
        name =  'fc1'
        )
    l_drop3 = lasagne.layers.DropoutLayer(
        incoming = l_fc1,
        p = 0.5,
        name = 'drop3',
        )        
    l_fc2 = lasagne.layers.DenseLayer(
        incoming = l_drop3,
        num_units = n_classes,
        W = lasagne.init.GlorotUniform(),
        nonlinearity = None,
        name = 'fc2'
        )
    return {'l_in':l_in, 'l_out':l_fc2}
    # reference https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
