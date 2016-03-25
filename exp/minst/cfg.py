import numpy as np

import lasagne
import lasagne.layers

import voxnet

lr_schedule = { 0: 0.001,
                60000: 0.0001,
                400000: 0.00005,
                600000: 0.00001,
                }

cfg = {'batch_size' : 64,
       'learning_rate' : lr_schedule,
       'reg' : 0.001,
       'momentum' : 0.8,
       'dims' : (28,28,1),
       'n_channels' : 1,
       'n_classes' : 10,
       'batches_per_chunk': 64, 
       'max_epochs' : 40,
       'max_jitter_ij' : 5,
       'max_jitter_k' : 0,
       'n_rotations' : 1,
       'checkpoint_every_nth' : 2000,
       }

def get_model():
    dims, n_channels, n_classes = tuple(cfg['dims']), cfg['n_channels'], cfg['n_classes']
    shape = (None, n_channels)+dims

    l_in = lasagne.layers.InputLayer(shape=shape)
    l_conv1 = voxnet.layers.Conv3dMMLayer(
            input_layer = l_in,
            num_filters = 8,
            filter_size = [8,8,1],
            border_mode = 'valid',
            strides = [1,1,1],
            W = voxnet.init.Prelu(),
            nonlinearity = voxnet.activations.leaky_relu_001,
            name =  'conv1'
        )
    l_pool1 = voxnet.layers.MaxPool3dLayer(
        input_layer = l_conv1,
        pool_shape = [2,2,1],
        name = 'pool1',
        )        
    l_drop1 = lasagne.layers.DropoutLayer(
        incoming = l_pool1,
        p = 0.4,
        name = 'drop1'
        )
    l_conv2 = voxnet.layers.Conv3dMMLayer(
            input_layer = l_drop1,
            num_filters = 16,
            filter_size = [5,5,1],
            border_mode = 'valid',
            W = voxnet.init.Prelu(),
            nonlinearity = voxnet.activations.leaky_relu_01,
            name =  'conv2'
        )      
    l_pool2 = voxnet.layers.MaxPool3dLayer(
        input_layer = l_conv2,
        pool_shape = [3,3,1],
        name = 'pool2',
        )
    l_drop2 = lasagne.layers.DropoutLayer(
        incoming = l_pool2,
        p = 0.5,
        name = 'drop2',
        )
    l_fc1 = lasagne.layers.DenseLayer(
        incoming = l_drop2,
        num_units = 128,
        W = lasagne.init.Normal(std=0.01),
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
        W = lasagne.init.Normal(std = 0.01),
        nonlinearity = None,
        name = 'fc2'
        )
    return {'l_in':l_in, 'l_out':l_fc2}
