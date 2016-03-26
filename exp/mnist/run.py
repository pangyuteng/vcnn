import os
import imp
    
import __vcnn__
import vcnn
from vcnn.utils import custom_logger

# logging
folder_path = os.path.dirname(__file__)
logger = custom_logger.new(folder_path,name=None)
logger.info('initiated')


# for training
class train_params:
    data_cls = 'Mnist'
    config_path = os.path.join(folder_path,'cfg.py')
    weights_fname = os.path.join(folder_path,'weights.npz')
    train_metrics_fname = os.path.join(folder_path,'metrics_train.jsonl')
    valid_metrics_fname = os.path.join(folder_path,'metrics_valid.jsonl')

class test_params:
    data_cls = train_params.data_cls
    out_fname = os.path.join(folder_path,'out_test.npz')
    config_path = train_params.config_path
    weights_fname = train_params.weights_fname

class report_params:
    metrics_fname = train_params.train_metrics_fname
    valid_metrics_fname = train_params.valid_metrics_fname
    train_acc_fname = None
    test_acc_fname = test_params.out_fname
    out_fname = os.path.join(folder_path,'report.html')        

if __name__ == '__main__':
    vcnn.utils.lsg.train(train_params)
    vcnn.utils.lsg.test(test_params)
    vcnn.utils.train_test_reports.main(report_params)
