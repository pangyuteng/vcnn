import os
import h5py
import numpy as np

def _remove(fname):
    if os.path.exists(fname):
        os.remove(fname)
        
class Writer(object):
    def __init__(self,fname,dims):
        if not isinstance(dims,tuple):
            raise TypeError('dims should be a tuple')
        _remove(fname)
        self.store = h5py.File(fname)
        self.X = self.store.create_dataset('X', (0,)+dims, maxshape=(None,)+dims)
        self.Y = self.store.create_dataset('Y', (0,1),'S10', maxshape=(None,1))

    def add(self,arr,name):
        new_shape_x = list(self.X.shape)
        new_shape_y = list(self.Y.shape)
        
        ind = new_shape_x[0]
        
        new_shape_x[0]+=1
        new_shape_y[0]+=1
        new_shape_x = tuple(new_shape_x)
        new_shape_y = tuple(new_shape_y)
        self.X.resize(new_shape_x)        
        self.Y.resize(new_shape_y)
        
        self.X[ind,:]=arr        
        self.Y[ind,:]=name.encode("ascii", "ignore")
      
    def close(self):
        self.store.close()

class Reader(object):
    def __init__(self, fname,random=False):
        self.store = h5py.File(fname,'r')
        self.shape = list(self.store['X'].shape)
        self._list = list(range(0,self.shape[0],1))
        if random:
            np.random.seed(seed=42)
            np.random.shuffle(self._list)          
        self._iter = iter(self._list)
            
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):   
        ind = self._iter.next()
        arr = self.store['X'][ind,:]
        name = self.store['Y'][ind,0]
        return arr, name

    def close(self):
        self.store.close()

