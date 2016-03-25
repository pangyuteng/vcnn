
import argparse
import logging

from path import Path
from scipy import ndimage

import numpy as np
from PIL import Image
from io import BytesIO as StringIO
from base64 import b64encode

class show_png():
    #https://github.com/Zulko/gizeh/blob/master/gizeh/gizeh.py 
    def __init__(self,data,type='L',format='png',normalize=False):
        self.data = data
        if normalize:
            self.data = np.round(255*(data-np.min(data))/(np.max(data)-np.min(data))).astype(np.uint8)
        else:
            self.data = np.round(255*self.data).astype(np.uint8)
        
        self.img = Image.fromarray(self.data, type)
        self.img=self.img.resize((30,30),Image.NEAREST)
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
    out = np.load(args.viz_weight_fname)
    
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

        f.write('<div>')
        f.write('<table><tr><td>')
               
        for dix in range(out['conv1.W'].shape[0]):
            xd = out['conv1.W'][dix]
            iloc = np.array(xd.shape)
            if len(iloc) ==4:
                for z in range(iloc[-1]):            
                    imgXY = show_png(xd[0,:,:,z],normalize=True).get_html_embed_code()                        
                    f.write(imgXY)
            else:
                imgXY = show_png(xd[0,:,:],normalize=True).get_html_embed_code()
                f.write(imgXY)
            f.write('{}<br>'.format(dix))
            
        f.write('</td></tr></table>')
        f.write('</div>')

        f.write('</body></html>')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('viz_weight_fname', type=Path)
    parser.add_argument('viz_fname', type=Path)
    args = parser.parse_args()
    main(args)    
