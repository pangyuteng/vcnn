import os
import imp
    
import __vcnn__
import vcnn
from vcnn.utils import custom_logger

# get data info
from vcnn.data import Mnist

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

class viz_params:
    data_cls = train_params.data_cls
    data_type = 'test'
    viz_out_fname = test_params.out_fname
    out_fname = os.path.join(folder_path,'viz_'+data_type+'_set.html')
    zoom = None 
    num_instances = 20

class params:
    #common params
    data_cls = 'Mnist'  # << defines dataset
    config_path = os.path.join(folder_path,'cfg.py') # << defines model and training configuration
    weights_fname = os.path.join(folder_path,'weights.npz')
    #training params
    train_metrics_fname = os.path.join(folder_path,'metrics_train.jsonl')
    valid_metrics_fname = os.path.join(folder_path,'metrics_valid.jsonl')
    #testing params
    out_fname = os.path.join(folder_path,'out_test.npz')


if __name__ == '__main__':
    #vcnn.utils.lsg.train(train_params)
    vcnn.utils.lsg.test(test_params)
    vcnn.utils.train_test_reports.main(report_params)
    vcnn.utils.lsg.viz(viz_params)

    # TODOS: switch to keras... or build below...
    #model = vcnn.utils.lsg.Model(params)
    #model.train()
    #model.test()
    #model.predict(inputs)
    #model.show_training_progress()
    #model.show_test_results()    
    #vcnn.utils.train_test_reports.main(report_params)
    #vcnn.utils.lsg.viz(viz_params)

