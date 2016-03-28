from __future__ import print_function

import os
import sys
import logging
import random
import numpy as np
from path import Path
import argparse
import numpy as np
import h5py

from ..utils import hdf5

from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from moviepy.editor import VideoFileClip
from scipy import ndimage


logger = logging.getLogger('mnist')


#---------------
class_id_to_name = {
    "0":'boxing',
    "1":'handclapping',
    "2":'handwaving',
    "3":'jogging',
    "4":'running',        
    "5":'walking',}
class_name_to_id = { v : k for k, v in class_id_to_name.items() }
class_names = set(class_id_to_name.values())
samples_per_group = 100
normalize = lambda x: (x-np.min(x))/(np.max(x)-np.min(x))
#---------------


def get_file_paths(categories,root_path):
    data_paths = {x:[] for x in categories}
    for category in categories:    
        folder = os.path.join(root_path,category)
        if not os.path.exists(folder):
            logging.error(DESCR)
            raise IOError('Data not found! Follow above instructions to get KTH Action dataset.')
            
        files = os.listdir(folder)
        for fname in files:
            file_path = os.path.join(folder,fname)
            if file_path.endswith('.avi'):
                data_paths[category].append(file_path)
    return data_paths

def avi2arr(fname,zoom=(0.5,1,0.75)):
    clip = VideoFileClip(fname)
    vid = np.array(list(clip.iter_frames()))[:,:,:,0]
    vid = ndimage.zoom(vid,zoom)
    return vid

class Writer(object):
    def __init__(self,fname,dims):
        if not isinstance(dims,tuple):
            raise TypeError('dims should be a tuple')
        hdf5._remove(fname)
        self.store = h5py.File(fname)
        self.X = self.store.create_dataset('X', (0,)+dims, maxshape=(None,)+dims)
        self.Y = self.store.create_dataset('Y', (0,1), maxshape=(None,1))

    def add(self,x,y):
        new_shape_x = list(self.X.shape)
        new_shape_y = list(self.Y.shape)
        
        ind = new_shape_x[0]
        
        new_shape_x[0]+=1
        new_shape_y[0]+=1
        new_shape_x = tuple(new_shape_x)
        new_shape_y = tuple(new_shape_y)
        self.X.resize(new_shape_x)        
        self.Y.resize(new_shape_y)        
        self.X[ind,:]=x     
        self.Y[ind,:]=y
      
    def close(self):
        self.store.close()
        
    '''
    Source: KTH Action Database
    Schuldt, Laptev and Caputo, Proc. ICPR'04, Cambridge, UK http://www.nada.kth.se/cvap/actions/
    '''

def _generate(train_path, valid_path, test_path,dims):
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
            
    logging.info('allocating train and testing set.')    
    # get file path for each class
    data_paths = get_file_paths(class_names,os.path.dirname(train_path))
    for k,v in data_paths.items():
        assert(len(v)==samples_per_group)
        
    # get random index for training/validation/testing data generation
    data_len = len(data_paths[list(class_names)[0]])
    rs = cross_validation.ShuffleSplit(data_len, n_iter=1,test_size=.2, random_state=0)
    for train_valid_index,test_index in rs:        
        _rs = cross_validation.ShuffleSplit(len(train_valid_index), n_iter=1, test_size=.2, random_state=0)
        for train_index,valid_index in _rs:
            pass
    # prepare for data generation
    data_index = {'train':train_index,'valid':valid_index,'test':test_index}
    records_key = ['train','valid','test']
    fnames = {'train':train_path,'valid':valid_path,'test':test_path}
    # for each data type (train/valid/test), create data.
    for data_type in records_key:  
        store = Writer(fnames[data_type],dims)
        logging.info('writing %s data %r' % (data_type,fnames[data_type]))
        for category in list(class_names):
            logging.info('category: %r' % category)
            for ind in data_index[data_type]:
                vid_path = data_paths[category][ind]
                vid = normalize(avi2arr(vid_path))
                vid_shape = list(vid.shape)
                x = np.zeros(dims)  
                # check for dimension 
                if vid_shape[0] >= dims[1]:
                    x[0]=vid[:dims[1],:,:]
                else:
                    x[0,:vid_shape[0],:,:]=vid
                    
                y = int(class_name_to_id[category])
                store.add(x,y)
        store.close()
        
from vcnn.conf import DATA_ROOT
class Ktha():
    class_id_to_name = class_id_to_name
    class_name_to_id = class_name_to_id
    data_path = os.path.join(DATA_ROOT,'ktha')
    train_path = os.path.join(data_path,'data_train.hdf5')
    valid_path = os.path.join(data_path,'data_valid.hdf5')
    test_path = os.path.join(data_path,'data_test.hdf5')    
    dims = (1,120,120,120)    
    @classmethod
    def generate(cls,force_gen=False):
        if force_gen is False:
            paths_of_interest = [cls.train_path,cls.valid_path,cls.test_path]
            if all([os.path.exists(x) for x in paths_of_interest]):
                logging.info('data found,no need to generate')
                return
        logging.info('generating data')
        _generate(cls.test_path,cls.valid_path,cls.train_path,cls.dims)
        
    @classmethod # todo. optimize.
    def get_dataset(cls): 
        cls.generate()
        store = h5py.File(cls.train_path,'r')
        X_train = store['X'][:]
        y_train = store['Y'][:]        
        store.close()
        
        store = h5py.File(cls.valid_path,'r')
        X_val = store['X'][:]
        y_val = store['Y'][:]
        store.close()
        
        store = h5py.File(cls.test_path,'r')
        X_test = store['X'][:]
        y_test = store['Y'][:]
        store.close()
        return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == '__main__':
    pass
