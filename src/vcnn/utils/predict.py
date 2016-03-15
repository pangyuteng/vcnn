import theano
import theano.tensor as T
import lasagne

from sklearn import metrics as skm
import logging
import imp
import traceback
import time

import numpy as np
import voxnet

import hdf5
#from . import hdf5
from rolling_window import rolling_window

def chestnorm(x):
    x = (x+400.)/(1500.)
    x[x<0] = 0.
    x[x>1] = 1.
    x=0.05+x*0.9
    return x
def get_cube(coord,img,lim,mask=None):
    n,p = lim
    cx,cy,cz = tuple(coord)
    return img[cx+n:cx+p,cy+n:cy+p,cz+n:cz+p]
    
def get_coords(npimg,lim):
    records = { 'test': []}
    logging.info('gathering coords')
    nlim = np.array([lim[1],lim[1],lim[1]])*2 #??
    plim = np.array(npimg.shape)-nlim*2 #?? too big of a borderr!
    coord_arr = np.column_stack(np.where(~np.isnan(npimg))).astype(np.int32)
    logging.info('coord shape %r, %r' % (coord_arr.shape))
    coord_arr = coord_arr[np.logical_and(coord_arr>nlim,coord_arr<plim).all(axis=1)]
    slice = 150
    print(slice)
    coord_arr = coord_arr[coord_arr[:,2]==slice]

    return coord_arr
def write(npimg_img_path,fname,cfg,pre_process=None):    
    dims = cfg.cfg['dims']
    lim = cfg.lim
    store = hdf5.Writer(fname,dims)
    npimg =np.load(npimg_img_path)        
    if pre_process:
        npimg = pre_process(npimg) 
    records = get_coords(npimg,lim,)
    
    for coord in records:                                
        name = '{:03d}.{:03d}.{:03d}'.format(*list(coord))
        arr = get_cube(coord,npimg,lim)
        store.add(arr,name)
        
def make_test_functions(cfg, model):
    l_out = model['l_out']
    batch_index = T.iscalar('batch_index')
    # bct01
    X = T.TensorType('float32', [False]*5)('X')
    y = T.TensorType('int32', [False]*1)('y')
    out_shape = lasagne.layers.get_output_shape(l_out)
    #log.info('output_shape = {}'.format(out_shape))

    batch_slice = slice(batch_index*cfg['batch_size'], (batch_index+1)*cfg['batch_size'])
    out = lasagne.layers.get_output(l_out, X)
    dout = lasagne.layers.get_output(l_out, X, deterministic=True)

    params = lasagne.layers.get_all_params(l_out)

    softmax_out = T.nnet.softmax( out )
    pred = T.argmax( dout, axis=1 )

    X_shared = lasagne.utils.shared_empty(5, dtype='float32')

    dout_fn = theano.function([X], dout)
    pred_fn = theano.function([X], pred)

    tfuncs = {'dout' : dout_fn,
             'pred' : pred_fn,
            }
    tvars = {'X' : X,
             'y' : y,
             'X_shared' : X_shared,
            }
    return tfuncs, tvars


def data_loader(config_module, chunk_size, npimg,):
    dims = config_module.cfg['dims']
    channels = config_module.cfg['n_channels']
    xc = np.zeros((chunk_size, channels,)+dims, dtype=np.float32)
    yc = []
    coord_arr = get_coords(npimg,config_module.lim,)
    view=rolling_window(npimg,dims)
    for ix, coord in enumerate(coord_arr):
        cix = ix % chunk_size       
        xc[cix,0,:,:,:] = view[coord[0],coord[1],coord[2],:,:,:].astype(np.float32)
        yc.append(coord)
        if len(yc) == chunk_size:
            yield (2.0*xc - 1.0, np.asarray(yc, dtype=np.int32))
            yc = []
            xc.fill(0)
    assert(len(yc)==0)
    
def main(args):

    # load config
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s| %(message)s')
    config_module = imp.load_source('config', args.config_path)
    cfg = config_module.cfg
    model = config_module.get_model()
    channels = config_module.cfg['n_channels']
    dims = config_module.cfg['dims']
    
    # load weight
    logging.info('Loading weights from {}'.format(args.weights_fname))
    voxnet.checkpoints.load_weights(args.weights_fname, model['l_out'])
        
    npimg =np.load(args.img_path)        
    npimg = chestnorm(npimg) 

    out_img = np.copy(npimg)
    out_img[:] = -1
    # predict
    tfuncs, tvars = make_test_functions(cfg, model)
    a=time.time()
    print(out_img.shape)
    try:

        for x_shared,coord_arr in data_loader(config_module, 64, npimg,):
            print(time.time()-a,'@')
            b=time.time()
            pred = np.argmax(tfuncs['dout'](x_shared),axis=1)
            print(time.time()-b,'!')
            a=time.time()
            assert(len(coord_arr)==len(pred))
            #indices = [np.ravel_multi_index(x,npimg.shape) for x in coord_arr]
            #out_img=np.insert(out_img,indices,pred)
            for n,coord in enumerate(coord_arr):
                out_img[coord[0],coord[1],coord[2]]=pred[n]
            print(np.sum(pred==0),np.sum(pred==1),np.sum(pred==2))
    except:
        traceback.print_exc()
    print(out_img.shape)
    # save out image
    np.save(args.out_img_path,out_img)
    # save to hr2...
    
class params:
    img_path=r"D:\SciSoft\codes\VCNN\data\lung\1.3.12.2.1107.5.1.4.24036.4.0.8285943131981741\10048\img.npy"
    img_in_path = r'D:\SciSoft\codes\VCNN\exp\1_lung\in_img.hdf5'
    config_path=r"D:\SciSoft\codes\VCNN\exp\1_lung\lung_cfg.py"
    weights_fname=r"D:\SciSoft\codes\VCNN\exp\1_lung\weights.npz"
    out_img_path = r'D:\SciSoft\codes\VCNN\exp\1_lung\out_img.npy'
    
if __name__=='__main__':

    #parser = argparse.ArgumentParser()
    #parser.add_argument('img_path', type=Path)    
    #parser.add_argument('config_path', type=Path)
    #parser.add_argument('weights_fname', type=Path, default='weights.npz')    
    #parser.add_argument('out_img_path', type=Path)
    #args = parser.parse_args()
    #main(args)
    start=time.time()
    main(params)
    end=time.time()
    print('delta',end-start)