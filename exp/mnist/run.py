import os
import imp
    
import __vcnn__
import vcnn
from vcnn.utils import custom_logger
# get data info
import vcnn.data as vcnndata

# logging
folder_path = os.path.dirname(__file__)
logger = custom_logger.new(folder_path,name=None)
logger.info('initiated')

# parameters for train/test model
class model_params:
    #common params
    data_cls = 'Mnist'  # << defines dataset
    config_path = os.path.join(folder_path,'cfg.py') # << defines model and training configuration
    weights_fname = os.path.join(folder_path,'weights.npz')
    #training params
    train_metrics_fname = os.path.join(folder_path,'metrics_train.jsonl')
    valid_metrics_fname = os.path.join(folder_path,'metrics_valid.jsonl')
    #testing params
    out_fname = os.path.join(folder_path,'out_test.npz')

# parameters for generating eval report
class report_params:
    metrics_fname = model_params.train_metrics_fname
    valid_metrics_fname = model_params.valid_metrics_fname
    train_acc_fname = None
    test_acc_fname = model_params.out_fname
    out_fname = os.path.join(folder_path,'report.html')        
    
# parameters for generating viz report
class viz_params:
    data_cls = model_params.data_cls
    data_type = 'test'
    viz_out_fname = model_params.out_fname
    out_fname = os.path.join(folder_path,'viz_'+data_type+'_set.html')
    zoom = None 
    num_instances = 20
    
if __name__ == '__main__':
    
    cls = getattr(vcnndata, model_params.data_cls)
    X_train, y_train, X_val, y_val, X_test, y_test = cls.get_dataset()
    model = vcnn.utils.lsg.Model(model_params)
    model.fit(X_train, y_train, X_val, y_val)
    model.evaluate(X_test, y_test)
    
    vcnn.utils.train_test_reports.main(report_params)
    vcnn.utils.lsg.viz(viz_params, X_test, y_test)

    # TODOS: switch to keras...