import os
import imp
    
import __vcnn__
import vcnn
from vcnn.utils import custom_logger
# logging
folder_path = os.path.dirname(__file__)
logger = custom_logger.new(folder_path,name=None)
logger.info('initiated')

# get data info
from vcnn.data import Minst2D

# for training
class train_params:    
    config_path = os.path.join(folder_path,'cfg.py')
    weights_fname = os.path.join(folder_path,'weights.npz')
    training_fname = Minst2D.train_path
    valid_fname = Minst2D.valid_path    
    metrics_fname = os.path.join(folder_path,'metrics.jsonl')
    valid_metrics_fname = os.path.join(folder_path,'metrics_valid.jsonl')

class test_params_trainset:
    config_path = train_params.config_path
    weights_fname = train_params.weights_fname
    testing_fname = Minst2D.train_path
    out_fname = os.path.join(folder_path,'out_train.npz')

class test_params_testset:
    config_path = train_params.config_path
    weights_fname = train_params.weights_fname
    testing_fname = Minst2D.test_path
    out_fname = os.path.join(folder_path,'out_test.npz')
    
class viz_params:
    cls_name = 'Minst2D'
    zoom = (4,4)
    viz_out_fname = test_params_testset.out_fname
    viz_data_fname = test_params_testset.testing_fname
    viz_fname = os.path.join(folder_path,'viz_test.html')
    num_instances = 20
    
class report_params:
    metrics_fname = train_params.metrics_fname
    valid_metrics_fname = train_params.valid_metrics_fname
    train_acc_fname = test_params_trainset.out_fname
    test_acc_fname = test_params_testset.out_fname
    out_fname = os.path.join(folder_path,'report.html')        
    
mail_info = {
    'title': 'results',
    'append': [
        {'attachment':report_params.out_fname},
        {'attachment':os.path.join(folder_path,'log.txt')},
    ]
}

if __name__ == '__main__':
    Minst2D.generate()
    vcnn.utils.train.main(train_params)
    vcnn.utils.test.main(test_params_trainset)    
    vcnn.utils.test.main(test_params_testset)    
    vcnn.utils.train_test_reports.main(report_params)
    vcnn.utils.viz.main(viz_params)
    vcnn.utils.send_mail(mail_info)
    
        