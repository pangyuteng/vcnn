

from __future__ import print_function

import sys
import os
import time
import imp
import logging

import numpy as np
import theano
import theano.tensor as T

import lasagne

import voxnet
from . import hdf5,viz_weights
from .. import data as vcnndata
from .lsg_viz import viz
logger = logging.getLogger('lsg')


def iterate_minibatches(inputs, targets, batchsize, shuffle=False,convert=True):
    assert len(inputs) == len(targets)
    if convert:
        inputs = inputs.astype(np.float32)
        targets = np.squeeze(targets.astype(np.int32))
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)     
        yield inputs[excerpt], targets[excerpt]


def _commons(args):
    config_module = imp.load_source('config', args.config_path)
    cfg = config_module.cfg    

    # Prepare Theano variables for inputs and targets
    input_var = cfg['input_T']('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    network = config_module.get_model(input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=cfg['learning_rate'], momentum=cfg['momentum'])

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)


    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    
    # third function for actual output.
    pred_fn = theano.function([input_var], [test_prediction])

    return cfg,network,train_fn,val_fn,pred_fn


def _train(args,cfg,network,train_fn,val_fn, X_train, y_train, X_val, y_val,):
        
    # Finally, launch the training loop.
    logger.info("Start training...")
    train_mlog = voxnet.metrics_logging.MetricsLogger(args.train_metrics_fname, reinitialize=True) 
    valid_mlog = voxnet.metrics_logging.MetricsLogger(args.valid_metrics_fname, reinitialize=True) 
    itr = 0
    # We iterate over epochs:
    for epoch in range(cfg['num_epochs']):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, cfg['batch_size'], shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            print(train_err)
            train_batches += 1
            itr+=train_batches            
            train_mlog.log(epoch=epoch, itr=itr, loss=train_err/train_batches, acc=0.0)

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch_val in iterate_minibatches(X_val, y_val, cfg['batch_size'], shuffle=False):
            inputs_val, targets_val = batch_val
            err, acc = val_fn(inputs_val, targets_val)
            print(err,acc)
            val_err += err
            val_acc += acc
            val_batches += 1            
        valid_mlog.log(epoch=epoch, itr=itr, loss=val_err/val_batches, acc=val_acc/val_batches)

        # Then we print the results for this epoch:
        logger.info("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, cfg['num_epochs'], time.time() - start_time))
        logger.info("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        logger.info("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        logger.info("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
            
        np.savez(args.weights_fname, *lasagne.layers.get_all_param_values(network))
        
def _test(args,cfg,val_fn,pred_fn, X_test, y_test):

    logger.info("Start testing...")
    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    yhat = []
    ygnd = []
    for batch in iterate_minibatches(X_test, y_test, cfg['batch_size'], shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)        
        test_err += err
        test_acc += acc
        test_batches += 1        
        ygnd.extend(targets)
        yhat.extend(np.squeeze(np.argmax(np.asarray(pred_fn(inputs)),axis=2).T))        

    logger.info("Final results:")
    logger.info("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    logger.info("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    if args.out_fname is not None:
        logger.info('saving predictions to {}'.format(args.out_fname))
        np.savez_compressed(args.out_fname, yhat=yhat, ygnd=ygnd)

        
class Model():
    def __init__(self,args):    
        self.args = args
        self.cfg,self.network,self.train_fn,self.val_fn,self.pred_fn =_commons(self.args)
        self._weights_loaded = False
        
    def fit(self,X_train, y_train, X_val, y_val):
        self._weights_loaded = False
        _train(self.args,self.cfg,self.network,self.train_fn,self.val_fn,X_train, y_train, X_val, y_val)        
        
    def _load_weights(self):
        with np.load(self.args.weights_fname) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]        	
            lasagne.layers.set_all_param_values(self.network, param_values)    
        self._weights_loaded = True
        
    def evaluate(self,X_test, y_test):
        self._load_weights()
        _test(self.args,self.cfg,self.val_fn,self.pred_fn, X_test, y_test)
        
    def predict(self,inputs,force_load=False):
        if self._weights_loaded is False or force_load is True:
            self._load_weights()        
        return np.squeeze(np.argmax(np.asarray(self.pred_fn(inputs)),axis=2).T)
        #^^^ # use flatten?
    def predict_proba(self,inputs):
        if self._weights_loaded is False or force_load is True:
            self._load_weights()
        raise NotImplementedError()