import os
import sys
import logging
import random
import numpy as np
from path import Path
import argparse

from ..utils import hdf5
from scipy import ndimage
from sklearn.cross_validation import train_test_split


#---------------
class_id_to_name = {
    "0": "ClassTrue",
    "1": "ClassFalse",
}
class_name_to_id = { v : k for k, v in class_id_to_name.items() }
class_names = set(class_id_to_name.values())
dims = [32,32,32]
def get_cube(id):
    cube = np.random.rand(*dims)*0.3    
    if id == 0:
        cube[0:16,:,:] = 0.9
    elif id == 1:
        cube[:,:,16:-1] = 0.2
    cube = ndimage.filters.gaussian_filter(cube,[0.5,0.5,0.5])
    cube[cube>0.9] = 0.9
    
    return cube
#---------------

def write(records,fname):    
    writer = hdf5.Writer(fname,tuple(dims))
    for id,_ in records:        
        name = '{:03d}.{:s}'.format(id, class_id_to_name[str(id)])
        arr = get_cube(id)                
        writer.add(arr, name)
    writer.close()


def _generate(test_path,valid_path,train_path):
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    np.random.seed(seed=42)
    test_size = 0.2
    Y = np.random.randint(0,2,300)
    _ = Y

    logging.info('allocating train and testing set.')
    _,_, y_train_valid, y_test = train_test_split(_, Y, test_size=test_size, random_state=42)
    _ = y_train_valid
    _,_ ,y_train, y_valid = train_test_split(_, y_train_valid, test_size=test_size, random_state=42)

    records = {'train': [],'valid':[], 'test': []}
    indices = {'train': y_train,'valid':y_valid,'test':y_test}
    path_dict = {'train': train_path,'valid':valid_path,'test':test_path}
    for data_type in sorted(list(records.keys())):
        for ind in indices[data_type]:
            records[data_type].append((ind,None))
        # just shuffle train set
        logging.info('Saving %r file' % data_type)
        random.shuffle(records[data_type])
        write(records[data_type],path_dict[data_type])
    
from vcnn.conf import DATA_ROOT    
class Simu():
    class_id_to_name = class_id_to_name
    class_name_to_id = class_name_to_id
    data_path = os.path.join(DATA_ROOT,'simu')
    train_path = os.path.join(data_path,'data_train.hdf5')
    test_path = os.path.join(data_path,'data_test.hdf5')
    valid_path = os.path.join(data_path,'data_valid.hdf5')
    
    @classmethod
    def generate(self,force_gen=False):
        if force_gen is False:
            if os.path.exists(self.train_path) and os.path.exists(self.test_path) and os.path.exists(self.valid_path):
                logging.info('data found,no need to generate')
                return
        logging.info('generating data')
        _generate(self.test_path,self.valid_path,self.train_path)
        
        

if __name__ == '__main__':
    pass