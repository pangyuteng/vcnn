import logging
import os
import sys
import numpy as np
from moviepy.editor import VideoFileClip
from scipy import ndimage
from sklearn import cross_validation

from vcnn.conf import DATA_ROOT    
from vcnn.utils import hdf5

DESCR = '''
    KTH Action Database
    Schuldt, Laptev and Caputo, Proc. ICPR'04, Cambridge, UK http://www.nada.kth.se/cvap/actions/
    Run the following:
    
    mkdir data
    cd data
    mkdir kth_action
    wget http://www.nada.kth.se/cvap/actions/walking.zip
    wget http://www.nada.kth.se/cvap/actions/jogging.zip
    wget http://www.nada.kth.se/cvap/actions/running.zip
    wget http://www.nada.kth.se/cvap/actions/boxing.zip
    wget http://www.nada.kth.se/cvap/actions/handwaving.zip
    wget http://www.nada.kth.se/cvap/actions/handclapping.zip
    unzip walking.zip -d walking
    unzip jogging.zip -d jogging
    unzip running.zip -d running
    unzip boxing.zip -d boxing
    unzip handwaving.zip -d handwaving
    unzip handclapping.zip -d handclapping
    
    '''
    
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
dims = (30,40,160)
samples_per_group = 100
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

def avi2arr(fname,zoom=(0.25,0.25,0.5)):
    clip = VideoFileClip(fname)
    vid = np.array(list(clip.iter_frames()))[:,:,:,0]
    vid = vid.swapaxes(0,1).swapaxes(1,2)
    vid = ndimage.zoom(vid,zoom)
    return vid
    
normalize = lambda x: 0.05+0.9*(x-np.min(x))/(np.max(x)-np.min(x))
def write(records,fname):
    writer = hdf5.Writer(fname,tuple(dims))
    for name,vid_path in records:
        name = '{:03d}.{:s}'.format(int(class_name_to_id[name]),name)
        vid = normalize(avi2arr(vid_path))
        vid_shape = list(vid.shape)
        arr = np.zeros(dims)
        
        if vid_shape[-1] >= dims[-1]:
            arr=vid[:,:,:dims[-1]]
        else:
            arr[:,:,:vid_shape[-1]]=vid
        writer.add(arr, name)
    writer.close()

def _generate(train_path,valid_path,test_path):
    # get file path for each class
    data_paths = get_file_paths(class_names,os.path.dirname(train_path))
    for k,v in data_paths.items():
        assert(len(v)==samples_per_group)
    # get random index for training/validation/testing data generation
    data_len = len(data_paths[list(class_names)[0]])
    rs = cross_validation.ShuffleSplit(data_len, n_iter=1,test_size=.33, random_state=0)
    for train_valid_index,test_index in rs:        
        _rs = cross_validation.ShuffleSplit(len(train_valid_index), n_iter=1, test_size=.5, random_state=0)
        for train_index,valid_index in _rs:
            pass
    # prepare for data generation
    data_index = {'train':train_index,'valid':valid_index,'test':test_index}
    records = {'train':[],'valid':[],'test':[]}
    fnames = {'train':train_path,'valid':valid_path,'test':test_path}
    # for each data type (train/valid/test), create data.
    for data_type in records.keys():
        logging.info('writing %s data %r' % (data_type,fnames[data_type]))
        index = data_index[data_type]
        for category in list(class_names):
            logging.info('category: %r' % category)
            for ind in index:
                records[data_type].append((category,data_paths[category][ind]))        
        write(records[data_type],fnames[data_type])

class KthAction():
    class_id_to_name = class_id_to_name
    class_name_to_id = class_name_to_id
    data_path = os.path.join(DATA_ROOT,'kth_action')
    train_path = os.path.join(data_path,'data_train.hdf5')
    valid_path = os.path.join(data_path,'data_valid.hdf5')
    test_path = os.path.join(data_path,'data_test.hdf5')    
    
    @classmethod
    def generate(self,force_gen=False):
        if force_gen is False:
            paths_of_interest = [self.train_path,self.valid_path,self.test_path]
            if all([os.path.exists(x) for x in paths_of_interest]):
                logging.info('data found,no need to generate')
                return
        logging.info('generating data')
        _generate(self.test_path,self.valid_path,self.train_path)
        
    

if __name__ == '__main__':
    a = KthAction()
    a.generate()