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
dims = [8,8,1]
#---------------


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
    test_size = 0.25    
    digits = datasets.load_digits()
    Y = digits.target.copy()
    X = digits.images.reshape(digits.images.shape[0],digits.images.shape[1]*digits.images.shape[2])
    
    logging.info('allocating train and testing set.')    
    X_train_valid,x_test,Y_train_valid,y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    x_train,x_valid, y_train, y_valid = train_test_split(X_train_valid,Y_train_valid, test_size=test_size, random_state=42)    

    records = {'train': [],'valid': [], 'test': []}
    ys = {'train': y_train,'valid': y_valid, 'test': y_test}
    xs = {'train': x_train,'valid': x_valid, 'test': x_test}
    paths = {'train': train_path,'valid': valid_path, 'test': test_path}
    for data_type in sorted(list(records.keys())):
        for n,y in enumerate(ys[data_type]):
            x = normalize(xs[data_type][n,:].reshape(dims))
            records[data_type].append((y,x))        
        # shuffle and save
        logging.info('Saving... %r ' % paths)
        random.shuffle(records[data_type])
        write(records[data_type],paths[data_type])

from vcnn.conf import DATA_ROOT
class Digits():
    class_id_to_name = class_id_to_name
    class_name_to_id = class_name_to_id
    data_path = os.path.join(DATA_ROOT,'digits')
    train_path = os.path.join(data_path,'data_train.hdf5')
    valid_path = os.path.join(data_path,'data_valid.hdf5')
    test_path = os.path.join(data_path,'data_test.hdf5')
    
    @classmethod
    def generate(self,force_gen=False):
        if force_gen is False:
            if os.path.exists(self.train_path) and os.path.exists(self.valid_path) and os.path.exists(self.test_path):
                logging.info('data found,no need to generate')
                return
        logging.info('generating data')
        _generate(self.train_path,self.valid_path,self.test_path)        
        
if __name__ == '__main__':
    pass
