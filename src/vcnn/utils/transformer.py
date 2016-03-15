import os
import theano
theano.config.floatX = 'float32'

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
def check_tensor(array, dtype=None, order=None, n_dim=None, copy=False):
    """Input validation on an array, or list.
    By default, the input is converted to an at least 2nd numpy array.
    Parameters
    ----------
    array : object
        Input object to check / convert.
    dtype : object
        Input type to check / convert.
    n_dim : int
        Number of dimensions for input array. If smaller, input array will be
        appended by dimensions of length 1 until n_dims is matched.
    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.
    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.
    Returns
    -------
    X_converted : object
        The converted and validated X.
    """
    array = np.array(array, dtype=dtype, order=order, copy=copy)
    if n_dim is not None:
        if len(array.shape) > n_dim:
            raise ValueError("Input array has shape %s, expected array with "
                             "%s dimensions or less" % (array.shape, n_dim))
        elif len(array.shape) < n_dim:
            array = array[[np.newaxis] * (n_dim - len(array.shape))]
    return array
    
class NeuralNetTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer/feature extractor for images using a neural network.
    ref  https://github.com/sklearn-theano/sklearn-theano/blob/master/sklearn_theano/feature_extraction/overfeat.py
    """
    def __init__(self, output_layers=[-1],
                 force_reshape=True,
                 transform_function = None,
                 batch_size=None):
        self.output_layers = output_layers
        self.force_reshape = force_reshape
        self.transform_function = transform_function
        self.batch_size = batch_size

        
    def transform(self, X):
        X = check_tensor(X, dtype=np.float32, n_dim=5)
        if self.batch_size is None:
            return self.transform_function(X)
        else:
            n_samples = X.shape[0]            
            for i in range(0, n_samples, self.batch_size):
                transformed_batch = self.transform_function(X[i:i + self.batch_size])
                # at first iteration, initialize output arrays to correct size
                if i == 0:
                    output = np.empty(X.shape[0], dtype=transformed_batch.dtype)
                print(i,self.batch_size)
                for transformed, out in zip(transformed_batch, output):
                    output[i:i + self.batch_size] = transformed
        return output

        
        
        #view=rolling_window(npimg,dims)
        
        
        
        
