import os
import time
import multiprocessing
import numpy as np
import logging
import tensorflow as tf
import char_keras_attention_lm as lm
from process_utils import UserConfig

logFormat = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logFormat)

# We use multiprocessing instead of thread to handle the Keras global state with multiple models
class ProcessUser(multiprocessing.Process):
    """
        Process user log for one day training the model then evaluate the day after
        In a separate thread. 
        Load the model and train it if available.
        Store the model for subsequent access.
    """

    def __init__(self, userConfig : UserConfig, day, num_days):
        """
            constructor

            user_dir: path to the user's directory, files are already processed per day
            model_dir: path to the user's model file
            day: day number to be used for training, eval will be day+1

        """
        multiprocessing.Process.__init__(self)
        self.userConfig = userConfig
        self.day = day
        self.num_days = num_days
        self.logger = logging.getLogger('ProcessUser')
        self.logger.setLevel(logging.INFO)
        
    def run(self):

        
        time.sleep(1)
        # file_name = "{}/{}.txt".format(self.userConfig.feat_dir, self.day)
        # logger.info("Processing file:", file_name)

        # load model if exist or create a new one
        char_lm = lm.KerasLM(self.userConfig)

        # train model
        self.logger.info("%s Training step...", self.userConfig.user_name)
        if char_lm.do_training(self.day):
            done_evaluation  = False  
            self.logger.info("%s Evaluation step...", self.userConfig.user_name)
            next_day = int(self.day)+1
            while not done_evaluation and next_day < self.num_days:
                done_evaluation = char_lm.do_evaluate(next_day)
                next_day +=1    


user_names_small = ['U8170', 'U3277', 'U8840', 'U7311', 'U1467', 'U1789', 'U8168', 'U1581', 'U7004', 'U9763']
user_names_moderate = ['U5254', 'U9407', 'U1592', 'U1723', 'U1106', 'U3406', 'U342', 'U1653', 
                'U20', 'U250', 'U1450', 'U1164', 'U86']
user_names_most_active = ['U12', 'U13', 'U24', 'U78', 'U207', 'U293', 'U453', 'U679', 'U1289', 'U1480']

user_names_short = ['U86', 'U342'] #['U1653', 'U1723']  # ['U24','U13','U1480'] #['U1653', 'U1723']
user_names = user_names_short

users_indir = '../data/users_feats'
users_lossdir = '../data/users_losses'
users_modeldir = '../data/users_models'
users_logidr = '../data/users_logs'
num_days = 27 #14 

# logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    # logging setup for consistency
    logger = logging.getLogger(__name__)

    logger.info("TensorFlow version: {}".format(tf.VERSION))

    if not os.path.exists(users_lossdir):
        os.makedirs(users_lossdir)

    if not os.path.exists(users_modeldir):
        os.makedirs(users_modeldir)
    
    if not os.path.exists(users_logidr):
        os.makedirs(users_logidr)

    for d in range(num_days):
        process_list = []
        for u in user_names:
            logger.info('Processing files for User: %s', u)
            userConfig = UserConfig()
            userConfig.user_name = u
            userConfig.feat_dir = '{0}/{1}/'.format(users_indir, u)
            userConfig.output_base_filepath = '{0}/{1}_losses'.format(users_lossdir, u)
            userConfig.model_filepath = '{0}/{1}_simple_lm.hdf5'.format(users_modeldir, u)
            userConfig.log_filepath = '{}/{}_log.txt'.format(users_logidr, u)

            process_user = ProcessUser(userConfig, d, num_days)
            process_list.append(process_user)

        logger.info(".... processing day: %d", d)
        for p in process_list:
            p.start()

        time.sleep(2)
        logger.info("... waiting for processing to finish for day: %d", d)
        for p in process_list:
            p.join()
        

