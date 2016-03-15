import numpy as np

# Lungs L:-400 W:1500
def chestnorm(x):
    x = (x+400.)/(1500.)
    x[x<0] = 0.
    x[x>1] = 1.
    x=0.05+x*0.9
    return x

def get_cube(coord,img,lim):
    n,p = lim
    cx,cy,cz = tuple(coord)
    return img[cx+n:cx+p,cy+n:cy+p,cz+n:cz+p]
    
def get_ortho(coord,img,lim):
    n,p = lim
    cx,cy,cz = tuple(coord)
    slice_xy = img[cx+n:cx+p,cy+n:cy+p,cz]
    slice_xz = img[cx+n:cx+p,cy,cz+n:cz+p]
    slice_yz = img[cx,cy+n:cy+p,cz+n:cz+p]
    
    patch = np.zeros(slice_xy.shape+(3,))
    patch[:,:,0]=slice_xy
    patch[:,:,1]=slice_xz
    patch[:,:,2]=slice_yz
    return patch