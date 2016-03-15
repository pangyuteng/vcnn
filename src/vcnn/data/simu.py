import os
import sys
import logging
import random
import numpy as np
from path import Path
import argparse

from ..utils import hdf5

from sklearn.cross_validation import train_test_split
dims = [32,32,32]
def get_cube(id):
    cube = np.random.rand(*dims)*0.3
    if id == 1:
        cube[0:16,:,:] = 0.9
    elif id == 2:
        cube[:,:,16:-1] = 0.2
    cube[cube>0.9] = 0.9
    return cube

def write(records,fname):    
    writer = hdf5.Writer(fname,tuple(dims))
    rot = 99
    for id,x in records:        
        name = '{:03d}.{:03d}.{:03d}'.format(id, id, rot)
        arr = get_cube(id)                
        writer.add(arr, name)
    writer.close()


def _generate(test_path, train_path):
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    np.random.seed(seed=42)
    test_size = 0.33
    Y = np.random.randint(1,3,3000)
    X = Y

    logging.info('allocating train and testing set.')
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)


    info = [
        {'train':[
            {'size':int(y_train.shape[0]) },
            {'category sample':[{1:int(np.sum(y_train == 1))},
                                {2:int(np.sum(y_train == 2))},
                               ]},
        ]
        },
        {'test':[
            {'size':int(y_test.shape[0]) },
            {'category sample':[{1:int(np.sum(y_test == 1))},
                                {2:int(np.sum(y_test == 2))},
                               ]},    
        ]
        }
    ]

    logging.info(str(info))

    records = {'train': [], 'test': []}
    for y,x in zip(y_train,x_train):
        records['train'].append((y,x))
        
    for y,x in zip(y_test,x_test):
        records['test'].append((y,x))

    # just shuffle train set
    logging.info('Saving train file')
    train_records = records['train']
    random.shuffle(train_records)
    write(train_records,train_path)

    # order test set by instance and orientation
    logging.info('Saving test file')
    test_records = records['test']
    write(test_records,test_path)
    
from vcnn.conf import DATA_ROOT    
class Simu():
    data_path = os.path.join(DATA_ROOT,'simu')
    train_path = os.path.join(data_path,'data_train.hdf5')
    test_path = os.path.join(data_path,'data_test.hdf5')
    
    @classmethod
    def generate(self,force_gen=False):
        if force_gen is False:
            if os.path.exists(self.train_path) and os.path.exists(self.test_path):
                logging.info('data found,no need to generate')
                return
        logging.info('generating data')
        _generate(self.test_path,self.train_path)
        
        
#---------------
class_id_to_name = {
    "1": "One",
    "2": "Two",
}
class_name_to_id = { v : k for k, v in class_id_to_name.items() }
class_names = set(class_id_to_name.values())
#---------------
if __name__ == '__main__':
    pass