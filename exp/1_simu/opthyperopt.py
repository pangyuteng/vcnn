# https://districtdatalabs.silvrback.com/parameter-tuning-with-hyperopt
# https://jaberg.github.io/hyperopt/
import os
import argparse
import imp
import time
import logging
import traceback

import numpy as np
from path import Path
import theano
import theano.tensor as T
import lasagne

import voxnet
import __vcnn__
from vcnn.utils import hdf5

import numpy as np
import lasagne
import lasagne.layers
        
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, STATUS_FAIL

def make_training_functions(cfg, model):
    l_out = model['l_out']
    batch_index = T.iscalar('batch_index')
    # bct01
    X = T.TensorType('float32', [False]*5)('X')
    y = T.TensorType('int32', [False]*1)('y')
    out_shape = lasagne.layers.get_output_shape(l_out)
    #log.info('output_shape = {}'.format(out_shape))

    batch_slice = slice(batch_index*cfg['batch_size'], (batch_index+1)*cfg['batch_size'])
    out = lasagne.layers.get_output(l_out, X)
    dout = lasagne.layers.get_output(l_out, X, deterministic=True)

    params = lasagne.layers.get_all_params(l_out)
    l2_norm = lasagne.regularization.regularize_network_params(l_out,
            lasagne.regularization.l2)
    if isinstance(cfg['learning_rate'], dict):
        learning_rate = theano.shared(np.float32(cfg['learning_rate'][0]))
    else:
        learning_rate = theano.shared(np.float32(cfg['learning_rate']))

    softmax_out = T.nnet.softmax( out )
    loss = T.cast(T.mean(T.nnet.categorical_crossentropy(softmax_out, y)), 'float32')
    pred = T.argmax( dout, axis=1 )
    error_rate = T.cast( T.mean( T.neq(pred, y) ), 'float32' )

    reg_loss = loss + cfg['reg']*l2_norm
    updates = lasagne.updates.momentum(reg_loss, params, learning_rate, cfg['momentum'])

    X_shared = lasagne.utils.shared_empty(5, dtype='float32')
    y_shared = lasagne.utils.shared_empty(1, dtype='float32')

    dout_fn = theano.function([X], dout)
    pred_fn = theano.function([X], pred)

    update_iter = theano.function([batch_index], reg_loss,
            updates=updates, givens={
            X: X_shared[batch_slice],
            y: T.cast( y_shared[batch_slice], 'int32'),
        })

    error_rate_fn = theano.function([batch_index], error_rate, givens={
            X: X_shared[batch_slice],
            y: T.cast( y_shared[batch_slice], 'int32'),
        })
    tfuncs = {'update_iter':update_iter,
             'error_rate':error_rate_fn,
             'dout' : dout_fn,
             'pred' : pred_fn,
            }
    tvars = {'X' : X,
             'y' : y,
             'X_shared' : X_shared,
             'y_shared' : y_shared,
             'batch_slice' : batch_slice,
             'batch_index' : batch_index,
             'learning_rate' : learning_rate,
            }
    return tfuncs, tvars

def jitter_chunk(src, cfg):
    dst = src.copy()
    if np.random.binomial(1, .2):
        dst[:, :, ::-1, :, :] = dst
    if np.random.binomial(1, .2):
        dst[:, :, :, ::-1, :] = dst
    max_ij = cfg['max_jitter_ij']
    max_k = cfg['max_jitter_k']
    shift_ijk = [np.random.random_integers(-max_ij, max_ij),
                 np.random.random_integers(-max_ij, max_ij),
                 np.random.random_integers(-max_k, max_k)]
    for axis, shift in enumerate(shift_ijk):
        if shift != 0:
            # beware wraparound
            dst = np.roll(dst, shift, axis+2)
    return dst

def data_loader(cfg, fname):

    dims = cfg['dims']
    chunk_size = cfg['batch_size']*cfg['batches_per_chunk']
    xc = np.zeros((chunk_size, cfg['n_channels'],)+dims, dtype=np.float32)
    reader = hdf5.Reader(fname)
    yc = []
    for ix, (x, name) in enumerate(reader):
        cix = ix % chunk_size
        xc[cix] = x.astype(np.float32)
        yc.append(int(name.split('.')[0])-1)
        if len(yc) == chunk_size:
            xc = jitter_chunk(xc, cfg)
            yield (2.0*xc - 1.0, np.asarray(yc, dtype=np.float32))
            yc = []
            xc.fill(0)
    if len(yc) > 0:
        # pad to nearest multiple of batch_size
        if len(yc)%cfg['batch_size'] != 0:
            new_size = int(np.ceil(len(yc)/float(cfg['batch_size'])))*cfg['batch_size']
            xc = xc[:new_size]
            xc[len(yc):] = xc[:(new_size-len(yc))]
            yc = yc + yc[:(new_size-len(yc))]

        xc = jitter_chunk(xc, cfg)
        yield (2.0*xc - 1.0, np.asarray(yc, dtype=np.float32))

class args:
    training_fname = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'data','simu','data_train.hdf5')    
    metrics_fname = 'metrics.jsonl'
    weights_fname = 'weights.npz'

def f(params):
    try:
        for k in sorted(list(params.keys())):
            print(k,params[k])
        lr_schedule = { 0: params['lr_0'],
                        60000: params['lr_60k'],
                        400000: params['lr_400k'],
                        600000: params['lr_600k'],
                        }

        cfg = {'batch_size' : params['batch_size'],
               'learning_rate' : lr_schedule,
               'reg' : params['reg'],
               'momentum' : params['momentum'],
               'dims' : (32, 32, 32),
               'n_channels' : 1,
               'n_classes' : 10,
               'batches_per_chunk': params['batches_per_chunk'],
               'max_epochs' : params['max_epochs'],
               'max_jitter_ij' : 2,
               'max_jitter_k' : 2,
               'n_rotations' : 12,
               'checkpoint_every_nth' : 4000,
               }
        

        dims, n_channels, n_classes = tuple(cfg['dims']), cfg['n_channels'], cfg['n_classes']
        shape = (None, n_channels)+dims

        l_in = lasagne.layers.InputLayer(shape=shape)
        l_conv1 = voxnet.layers.Conv3dMMLayer(
                input_layer = l_in,
                num_filters = params['num_filters'],
                filter_size = [5,5,5],
                border_mode = 'valid',
                strides = [2,2,2],
                W = voxnet.init.Prelu(),
                nonlinearity = voxnet.activations.leaky_relu_01,
                name =  'conv1'
            )
        l_drop1 = lasagne.layers.DropoutLayer(
            incoming = l_conv1,
            p = params['drop1p'],
            name = 'drop1'
            )
        l_conv2 = voxnet.layers.Conv3dMMLayer(
            input_layer = l_drop1,
            num_filters = params['num_filters'],
            filter_size = [3,3,3],
            border_mode = 'valid',
            W = voxnet.init.Prelu(),
            nonlinearity = voxnet.activations.leaky_relu_01,
            name = 'conv2'
            )
        l_pool2 = voxnet.layers.MaxPool3dLayer(
            input_layer = l_conv2,
            pool_shape = [2,2,2],
            name = 'pool2',
            )
        l_drop2 = lasagne.layers.DropoutLayer(
            incoming = l_pool2,
            p = params['drop2p'],
            name = 'drop2',
            )
        l_fc1 = lasagne.layers.DenseLayer(
            incoming = l_drop2,
            num_units = params['num_units'],
            W = lasagne.init.Normal(std=0.01),
            name =  'fc1'
            )
        l_drop3 = lasagne.layers.DropoutLayer(
            incoming = l_fc1,
            p = params['drop3p'],
            name = 'drop3',
            )
        l_fc2 = lasagne.layers.DenseLayer(
            incoming = l_drop3,
            num_units = n_classes,
            W = lasagne.init.Normal(std = 0.01),
            nonlinearity = None,
            name = 'fc2'
            )
        
        model = {'l_in':l_in, 'l_out':l_fc2}

        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s| %(message)s')
        logging.info('Metrics will be saved to {}'.format(args.metrics_fname))
        mlog = voxnet.metrics_logging.MetricsLogger(args.metrics_fname, reinitialize=True)

        logging.info('Compiling theano functions...')
        tfuncs, tvars = make_training_functions(cfg, model)

        logging.info('Training...')
        itr = 0
        last_checkpoint_itr = 0
        loader = (data_loader(cfg, args.training_fname))
        for epoch in xrange(cfg['max_epochs']):
            loader = (data_loader(cfg, args.training_fname))

            for x_shared, y_shared in loader:
                num_batches = len(x_shared)//cfg['batch_size']
                tvars['X_shared'].set_value(x_shared, borrow=True)
                tvars['y_shared'].set_value(y_shared, borrow=True)
                lvs,accs = [],[]
                for bi in xrange(num_batches):
                    lv = tfuncs['update_iter'](bi)
                    lvs.append(lv)
                    acc = 1.0-tfuncs['error_rate'](bi)
                    accs.append(acc)
                    itr += 1
                loss, acc = float(np.mean(lvs)), float(np.mean(acc))
                logging.info('epoch: {}, itr: {}, loss: {}, acc: {}'.format(epoch, itr, loss, acc))
                mlog.log(epoch=epoch, itr=itr, loss=loss, acc=acc)
                
                if isinstance(cfg['learning_rate'], dict) and itr > 0:
                    keys = sorted(cfg['learning_rate'].keys())
                    new_lr = cfg['learning_rate'][keys[np.searchsorted(keys, itr)-1]]
                    lr = np.float32(tvars['learning_rate'].get_value())
                    if not np.allclose(lr, new_lr):
                        logging.info('decreasing learning rate from {} to {}'.format(lr, new_lr))
                        tvars['learning_rate'].set_value(np.float32(new_lr))
                if itr-last_checkpoint_itr > cfg['checkpoint_every_nth']:
                    voxnet.checkpoints.save_weights(args.weights_fname, model['l_out'],
                                                    {'itr': itr, 'ts': time.time()})
                    last_checkpoint_itr = itr


        logging.info('training done')
        voxnet.checkpoints.save_weights(args.weights_fname, model['l_out'],
                                        {'itr': itr, 'ts': time.time()})
        return {'loss': loss, 'status': STATUS_OK}
    except:
        traceback.print_exc()
        return {'loss': np.nan, 'status': STATUS_FAIL}
        

if __name__=='__main__':   
    fspace = {
        'lr_0': hp.uniform('lr_0', 0.0001, 0.01),
        'lr_60k': hp.uniform('lr_60k', 0.0001, 0.01),
        'lr_400k': hp.uniform('lr_400k', 0.0001, 0.01),
        'lr_600k': hp.uniform('lr_600k', 0.0001, 0.01),
        'reg': hp.uniform('reg',0.0001,0.01),
        'momentum': hp.choice('momentum',[0.3,0.5,0.7,0.9]),
        'max_epochs': hp.choice('max_epochs',[3]),
        'drop1p': hp.uniform('drop1p',0.1,0.9),
        'drop2p': hp.uniform('drop2p',0.1,0.9),
        'drop3p': hp.uniform('drop3p',0.1,0.9),
        'num_filters': hp.choice('num_filters',[4,16,32]),
        'num_units': hp.choice('num_units',[16,128]),        
        'batches_per_chunk': hp.choice('batches_per_chunk',[32]),
        'batch_size': hp.choice('batch_size',[16]),
    }

    trials = Trials()
    best = fmin(fn=f, space=fspace, algo=tpe.suggest, max_evals=50, trials=trials)
    print('best',best)