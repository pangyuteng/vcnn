import os
import sys
import logging
import random
import numpy as np
from path import Path
import argparse
import numpy as np

from ..utils import hdf5

from sklearn.cross_validation import train_test_split
from sklearn import datasets

#---------------
class_id_to_name = {
    "0": "Zero",
    "1": "One",
    "2": "Two",
    "3": "Three",
    "4": "Four",
    "5": "Five",
    "6": "Six",
    "7": "Seven",
    "8": "Eight",
    "9": "Nine",
}

class_name_to_id = { v : k for k, v in class_id_to_name.items() }
class_names = set(class_id_to_name.values())
dims = [28,28,1]
#---------------

import os
import struct
import array
import numpy

def read(digits, dataset="training", path="."):
    """

    Source: http://g.sweyla.com/blog/2012/mnist-numpy/
    MNIST: http://yann.lecun.com/exdb/mnist/
    **Parameters**
        :digits: list; digits we want to load
        :dataset: string; 'training' or 'testing'
        :path: string; path to the data set files
    """

    if dataset is "train":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "test":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    struct.unpack(">II", flbl.read(8))
    lbl = array.array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    _, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array.array("B", fimg.read())
    fimg.close()

    ind = [k for k in xrange(size) if lbl[k] in digits]
    N = len(ind)

    images = numpy.zeros((N, rows, cols), dtype=numpy.uint8)
    labels = numpy.zeros((N, 1), dtype=numpy.int8)
    for i in xrange(len(ind)):
        images[i] = numpy.array(img[ind[i]*rows*cols:
                                (ind[i]+1)*rows*cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

def write(records,fname):    
    writer = hdf5.Writer(fname,tuple(dims))
    for id,arr in records:        
        name = '{:03d}.{:s}'.format(id, class_id_to_name[str(id)])              
        writer.add(arr, name)
    writer.close()

normalize = lambda x: 0.05+0.9*(x-np.min(x))/(np.max(x)-np.min(x))

def _generate(train_path, valid_path, test_path):
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    logging.info('load sklearn digits dataset')
    np.random.seed(seed=42)
    test_size = 0.2
    digits = list(range(10))
    x_test,y_test = read(digits, dataset='test', path=os.path.dirname(test_path))
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])
    X_tv,Y_tv = read(digits, dataset='train', path=os.path.dirname(train_path))   
    
    X_tv = X_tv.reshape(X_tv.shape[0],X_tv.shape[1]*X_tv.shape[2])
    
    logging.info('allocating train and testing set.')    
    x_train,x_valid, y_train, y_valid = train_test_split(X_tv,Y_tv, test_size=test_size, random_state=42)    

    records = {'train': [],'valid': [], 'test': []}
    ys = {'train': y_train,'valid': y_valid, 'test': y_test}
    xs = {'train': x_train,'valid': x_valid, 'test': x_test}
    paths = {'train': train_path,'valid': valid_path, 'test': test_path}
    for data_type in sorted(list(records.keys())):
        for n,y in enumerate(ys[data_type]):
            x = normalize(xs[data_type][n,:].reshape(dims))
            records[data_type].append((y[0],x))        
        # shuffle and save
        logging.info('Saving... %r ' % paths)
        random.shuffle(records[data_type])
        write(records[data_type],paths[data_type])

from vcnn.conf import DATA_ROOT
class Minst():
    class_id_to_name = class_id_to_name
    class_name_to_id = class_name_to_id
    data_path = os.path.join(DATA_ROOT,'minst')
    train_path = os.path.join(data_path,'data_train.hdf5')
    valid_path = os.path.join(data_path,'data_valid.hdf5')
    test_path = os.path.join(data_path,'data_test.hdf5')
    
    @classmethod
    def generate(self,force_gen=False):
        if force_gen is False:
            if os.path.exists(self.train_path) and os.path.exists(self.test_path) and os.path.exists(self.valid_path):
                logging.info('data found,no need to generate')
                return
        print('download and unzip to data/minst  http://yann.lecun.com/exdb/mnist/')
        logging.info('http://yann.lecun.com/exdb/mnist/')
        logging.info('generating data')
        _generate(self.train_path,self.valid_path,self.test_path)        
        
if __name__ == '__main__':
    # Minst.generate()
    pass 
