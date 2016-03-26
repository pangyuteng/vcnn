import logging
logger = logging.getLogger('report')

import cStringIO as StringIO
import argparse
import imp
import time
import matplotlib
matplotlib.use('Agg')
import seaborn
seaborn.set_style('whitegrid')

import numpy as np
import pandas as pd
from sklearn import metrics

import matplotlib.pyplot as pl
from path import Path

from matplotlib import rcParams

import voxnet

def get_recs(fname):
    recs = list(voxnet.metrics_logging.read_records(fname))    
    stamps = [r['_stamp'] for r in recs]
    df = pd.DataFrame(recs, index=stamps)
    df['loss'] = df['loss'].astype(np.float)
    df['acc'] = df['acc'].astype(np.float)
    df = df.dropna(axis=0)
    y_axis = 'epoch'
    acc = df['acc'].sort_index()
    acc.index = df.sort_index()[y_axis]
    loss = df['loss'].sort_index()
    loss.index = df.sort_index()[y_axis]
    return {'acc':acc,'loss':loss}
    
    
def set_matplotlib_params():
    rcParams['axes.labelsize'] = 12
    rcParams['xtick.labelsize'] = 12
    rcParams['ytick.labelsize'] = 12
    rcParams['legend.fontsize'] = 12
    golden_ratio = (np.sqrt(5)-1.0)/2.0
    fig_width_in = 8.
    fig_height_in = fig_width_in * golden_ratio
    rcParams['figure.figsize'] = [fig_width_in, fig_height_in]


def main(args):

    set_matplotlib_params()
    if args.train_acc_fname is not None:
        train_result = np.load(args.train_acc_fname)
        y_true = train_result['ygnd']
        y_pred = train_result['yhat']
        train_scores = {
            'accuracy':metrics.accuracy_score(y_true, y_pred),
            'precision':metrics.precision_score(y_true, y_pred),
            'confusion matrix': metrics.confusion_matrix(y_true, y_pred)/float(len(y_true)),
            'sample size':len(y_pred)
            }
        
    if args.test_acc_fname is not None:
        test_result = np.load(args.test_acc_fname)
        y_true = test_result['ygnd']
        y_pred = test_result['yhat']
        test_scores = {
            'accuracy':metrics.accuracy_score(y_true, y_pred),
            'precision':metrics.precision_score(y_true, y_pred),
            'confusion matrix': metrics.confusion_matrix(y_true, y_pred)/float(len(y_true)),
            'sample size':len(y_pred)
            }
            
    test_recs = get_recs(args.metrics_fname)
    if args.valid_metrics_fname:
        valid_recs = get_recs(args.valid_metrics_fname)

    smoothing_window = 15

    with open(args.out_fname, 'w') as page:
        page.write('<html><head></head><body>')
        page.write('<h1>Training report</h1>')
        page.write('<p>{}</p>'.format(time.ctime()))
        page.write('<h2>Loss</h2>')
        fig = pl.figure()
        test_recs['loss'].plot(label='raw test',linewidth=0.5)
        pd.rolling_mean(test_recs['loss'], smoothing_window).plot(label='smoothed test')
        if args.valid_metrics_fname:
            valid_recs['loss'].plot(label='raw valid',linewidth=0.5)
            pd.rolling_mean(valid_recs['loss'], smoothing_window).plot(label='smoothed valid')
        
        pl.xlabel('Iter')
        pl.ylabel('Loss')
        pl.legend()
        fig.tight_layout(pad=0.1)
        sio = StringIO.StringIO()
        pl.savefig(sio, format='svg')
        page.write(sio.getvalue())

        page.write('<h2>Accuracy</h2>')
        fig = pl.figure()
        test_recs['acc'].plot(label='raw test',linewidth=0.5)
        if args.valid_metrics_fname:
            valid_recs['acc'].plot(label='raw valid',linewidth=1)
        
        pl.xlabel('Iter')
        pl.ylabel('Accuracy')
        pl.legend()        
        fig.tight_layout(pad=0.1)
        sio = StringIO.StringIO()
        pl.savefig(sio, format='svg')
        page.write(sio.getvalue())
        
        if args.train_acc_fname is not None:
            page.write('<h2>training - confusion matrix</h2>')
            page.write('<p>sample size {}</p>'.format(train_scores['sample size']))
            page.write('<p>accuracy {}</p>'.format(train_scores['accuracy']))

            fig = pl.figure()
            ax = pl.subplot(111)
            pl.imshow(train_scores['confusion matrix'],interpolation='nearest',cmap='gray')
            ax.set_aspect(1)
            conf_arr = train_scores['confusion matrix']
            width, height = conf_arr.shape

            for x in xrange(width):
                for y in xrange(height):
                    ax.annotate('{:1.2f}'.format(conf_arr[x][y]), xy=(y, x), 
                                horizontalalignment='center',
                                verticalalignment='center',color='red')
            
            pl.colorbar()    
            pl.xlabel('Predicted')
            pl.ylabel('Ground')
            sio = StringIO.StringIO()
            pl.savefig(sio, format='svg')
            page.write(sio.getvalue())
        
        if args.test_acc_fname is not None:
            page.write('<h1>Test report</h1>')
            page.write('<h2>testing - confusion matrix</h2>')
            page.write('<p>sample size {}</p>'.format(test_scores['sample size']))
            page.write('<p>accuracy {}</p>'.format(test_scores['accuracy']))
            fig = pl.figure()
            ax = pl.subplot(111)
            pl.imshow(test_scores['confusion matrix'],interpolation='nearest',cmap='gray')
            ax.set_aspect(1)
            conf_arr = test_scores['confusion matrix']
            width, height = conf_arr.shape

            for x in xrange(width):
                for y in xrange(height):
                    ax.annotate('{:1.2f}'.format(conf_arr[x][y]), xy=(y, x), 
                                horizontalalignment='center',
                                verticalalignment='center',color='red')
            pl.colorbar()    
            pl.xlabel('Predicted')
            pl.ylabel('Ground')
            sio = StringIO.StringIO()
            pl.savefig(sio, format='svg')
            page.write(sio.getvalue())        
        
        page.write('</body></html>')
        
if __name__ == '__main__':
    logger.info('report initiated...')
    parser = argparse.ArgumentParser()
    parser.add_argument('metrics_fname', type=Path)
    parser.add_argument('valid_metrics_fname', type=Path,default=None)
    parser.add_argument('train_acc_fname', type=Path)
    parser.add_argument('test_acc_fname', type=Path)
    parser.add_argument('out_fname', type=Path)
    args = parser.parse_args()
    main(args)