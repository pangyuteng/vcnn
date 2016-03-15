import os
import logging.config

# setup the logger
def new(folder_path,name=None):
    logger = logging.getLogger(name)
    hdlr = logging.FileHandler(os.path.join(folder_path,'log.txt'),mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(lineno)s - %(message)s')
    hdlr.suffix = "%Y-%m-%d"
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger