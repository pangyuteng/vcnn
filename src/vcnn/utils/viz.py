
import argparse
import logging
logger = logging.getLogger('viz')

from path import Path
import numpy as np

from . import hdf5

import numpy as np
from PIL import Image
from io import BytesIO as StringIO
from base64 import b64encode
from scipy import ndimage
from .. import data as vcnndata

class show_png():
    #https://github.com/Zulko/gizeh/blob/master/gizeh/gizeh.py 
    def __init__(self,data,type='L',format='png',normalize=False):
        self.data = data
        if normalize:
            self.data = np.round(255*(data-np.min(data))/(np.max(data)-np.min(data))).astype(np.uint8)
        else:
            self.data = np.round(255*self.data).astype(np.uint8)
        
        self.img = Image.fromarray(self.data, type)
        #self.img=self.img.resize((30,30),Image.NEAREST)
        self.format = 'png'

    def write_to_png(self, fileobject, y_origin="top"):
        self.img.save(fileobject,self.format)
        
    def get_html_embed_code(self, y_origin="top"):
        """ Returns an html code containing all the PNG data of the surface. """
        png_data = self._repr_png_()
        data = b64encode(png_data).decode('utf-8')
        return "<img  src='data:image/png;base64,%s'>"%(data)

    def ipython_display(self, y_origin="top"):
        """ displays the surface in the IPython notebook.
        Will only work if surface.ipython_display() is written at the end of one
        of the notebook's cells.
        """

        from IPython.display import HTML
        return HTML(self.get_html_embed_code(y_origin=y_origin))

    def _repr_html_(self):
        return self.get_html_embed_code()

    def _repr_png_(self):
        """ Returns the raw PNG data to be displayed in the IPython notebook"""
        data = StringIO()
        self.write_to_png(data)
        return data.getvalue()
        
def main(args):
    reader = hdf5.Reader(args.viz_data_fname,random=False)
    out = np.load(args.viz_out_fname)
    cls = getattr(vcnndata, args.cls_name)
    
    yhat = out['yhat']
    ygnd = out['ygnd']

    css = """
    html { margin: 0 }
    body {
        background:#fff;
        color:#000;
        font:75%/1.5em Helvetica, "DejaVu Sans", "Liberation sans", "Bitstream Vera Sans", sans-serif;
        position:relative;
    }
    /*dt { font-weight: bold; float: left; clear; left }*/
    div { padding: 10px; width: 80%; margin: auto }
    img { border: 1px solid #eee }
    dl { margin:0 0 1.5em; }
    dt { font-weight:700; }
    dd { margin-left:1.5em; }
    table {
        border-collapse:collapse;
        border-spacing:0;
        margin:0 0 1.5em;
        padding:0;
    }
    td { padding:0.333em;
        vertical-align:middle;
    }
    }"""

    with open(args.viz_fname, 'w') as f:
        f.write('<html><head><style>')
        f.write(css)
        f.write('</style></head>')
        f.write('<body>')


        display_ix = np.random.randint(0, len(ygnd), args.num_instances)
        
        xds, yds = [], []
        for ix, (xd, yd) in enumerate(reader):
            if ix in display_ix:
                dix = ix
                if args.zoom:
                    xd = ndimage.interpolation.zoom(xd,args.zoom,mode='nearest')
                iloc = np.round(np.array(xd.shape)/2)
                if len(iloc) == 3:
                    imgXY = show_png(xd[:,:,iloc[2]],normalize=True).get_html_embed_code()
                    imgXZ = show_png(xd[:,iloc[1],:],normalize=True).get_html_embed_code()
                    imgYZ = show_png(xd[iloc[0],:,:],normalize=True).get_html_embed_code()
                elif len(iloc) ==2:
                    imgXY = show_png(xd[:,:],normalize=True).get_html_embed_code()                                    
                f.write('<div>')
                f.write('<table><tr><td>')
                if len(iloc) == 3:                
                    f.write(imgXY)
                    f.write(imgXZ)
                    f.write(imgYZ)
                elif len(iloc) == 2:
                    f.write(imgXY)
                f.write('</td>')
                f.write('<td>')
                f.write('<dl><dt>Instance:</dt><dd>{}</dd>'.format(yd))
                f.write('<dt>Predicted label:</dt><dd>{}</dd>'.format(cls.class_id_to_name[str(yhat[dix])]))
                f.write('<dt>True label:</dt><dd>{}</dd></dl>'.format(cls.class_id_to_name[str(ygnd[dix])]))
                f.write('</td></tr></table>')
                f.write('</div>')
                xds.append(xd)
                yds.append(yd)

        f.write('</body></html>')

if __name__ == '__main__':
    logger.info('viz initiated...')
    parser = argparse.ArgumentParser()
    parser.add_argument('cls_name', type=str)
    parser.add_argument('viz_out_fname', type=Path)
    parser.add_argument('viz_data_fname', type=Path)
    parser.add_argument('viz_fname', type=Path)
    parser.add_argument('zoom', type=Tuple, default=None)
    parser.add_argument('--num_instances', type=int, default=10)
    args = parser.parse_args()
    main(args)    
