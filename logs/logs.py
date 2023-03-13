import logging 
import os
import time
import sys

def logger_init(log_filename='moniter',
                log_level=logging.DEBUG,
                log_dir='./',
                only_file=False):
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir, log_filename+'_'+time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time())) + '.txt')
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    if only_file:
        logging.basicConfig(filename=log_path,
                            level=log_level,
                            format=formatter,
                            datefmt='%Y-%d-%m %H:%M:%S')
    else:
        logging.basicConfig(level=log_level,
                            format=formatter,
                            datefmt='%Y-%d-%m %H:%M:%S',
                            handlers=[logging.FileHandler(log_path),
                                      logging.StreamHandler(sys.stdout)]
                            )


if __name__ == '__main__':
    print (time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time())))