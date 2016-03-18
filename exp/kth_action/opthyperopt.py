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
import pickle

from vcnn.utils import train, test

class args:
    training_fname = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'data','kth_action','data_train.hdf5')    
    validate_fname = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'data','kth_action','data_train.hdf5')        
    metrics_fname = 'metrics.jsonl'
    weights_fname = 'weights.npz'

def f(params):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s| %(message)s')
    logging.info('Metrics will be saved to {}'.format(args.metrics_fname))
    mlog = voxnet.metrics_logging.MetricsLogger(args.metrics_fname, reinitialize=True)
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
               'dims' : (30,40,160),
               'n_channels' : 1,
               'n_classes' : 10,
               'batches_per_chunk': params['batches_per_chunk'],
               'max_epochs' : params['max_epochs'],
               'max_jitter_ij' : 2,
               'max_jitter_k' : 2,
               'n_rotations' : 1,
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

       
        logging.info('Compiling theano functions...')
        tfuncs, tvars = train.make_training_functions(cfg, model)

        logging.info('Training...')
        itr = 0
        last_checkpoint_itr = 0
        loader = (train.data_loader(cfg, args.training_fname))
        for epoch in xrange(cfg['max_epochs']):
            loader = (train.data_loader(cfg, args.training_fname))

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
                                        
                                        
        logging.info('Loading weights from {}'.format(args.weights_fname))
        voxnet.checkpoints.load_weights(args.weights_fname, model['l_out'])

        loader = (train.data_loader(cfg, args.validate_fname))

        tfuncs, tvars = train.make_test_functions(cfg, model)

        yhat, ygnd = [], []
        for x_shared, y_shared in loader:
            pred = np.argmax(np.sum(tfuncs['dout'](x_shared), 0))
            yhat.append(pred)
            ygnd.append(y_shared[0])
            
        assert(len(yhat)==len(ygnd))

        yhat = np.asarray(yhat, dtype=np.int)
        ygnd = np.asarray(ygnd, dtype=np.int)

        acc = np.mean(yhat==ygnd).mean()
        loss = 1.0-acc
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
        'reg': hp.uniform('reg',0.0001,0.5),
        'momentum': hp.uniform('momentum',0.01,0.99),
        'max_epochs': hp.choice('max_epochs',[3,4]),#[450,100,500]),
        'drop1p': hp.uniform('drop1p',0.1,0.9),
        'drop2p': hp.uniform('drop2p',0.1,0.9),
        'drop3p': hp.uniform('drop3p',0.1,0.9),
        'num_filters': hp.choice('num_filters',[16,32,64,128,256,512]),
        'num_units': hp.choice('num_units',[16,32,64,128,256,512,1024]),        
        'batches_per_chunk': hp.choice('batches_per_chunk',[32,64]),
        'batch_size': hp.choice('batch_size',[16]),
    }

    trials = Trials()
    best = fmin(fn=f, space=fspace, algo=tpe.suggest, max_evals=4, trials=trials)
    print('best',best)    
    with open('trials.pkl','wb') as f:
        pickle.dump({'trials':trials,'best':best},f)