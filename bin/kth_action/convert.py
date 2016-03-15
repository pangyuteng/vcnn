import os
import sys
import numpy as np
from moviepy.editor import VideoFileClip
from scipy import ndimage

categories = ['walking','handwaving']
root = os.path.dirname(os.path.abspath(__file__))
data_paths = {x:[] for x in categories}
for category in categories:    
    print(category)
    folder = os.path.join(root,category)
    files = os.listdir(folder)
    for fname in files:
        file_path = os.path.join(folder,fname)
        if file_path.endswith('.avi'):
            data_paths[category].append(file_path)
        
def avi2arr(fname,zoom=(0.25,0.25,0.25)):
    clip = VideoFileClip(fname)
    vid=np.array(list(clip.iter_frames()))[:,:,:,0]
    vid =vid.swapaxes(0,1).swapaxes(1,2)
    vid = scipy.ndimage.zoom(vid,zoom)

def write(records,fname):    
    writer = hdf5.Writer(fname,tuple(dims))
    rot = 99
    for id,x in records:        
        name = '{:03d}.{:03d}.{:03d}'.format(id, id, rot)
        arr = get_cube(id)                
        writer.add(arr, name)
    writer.close()