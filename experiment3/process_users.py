import os
import time
import multiprocessing
import numpy as np
import logging
import tensorflow as tf
import char_keras_lm as lm
from process_utils import UserConfig

logging.basicConfig(level=logging.INFO)

# We use multiprocessing instead of thread to handle the Keras global state with multiple models
class ProcessUser(multiprocessing.Process):
    """
        Process user log for one day training the model then evaluate the day after
        In a separate thread. 
        Load the model and train it if available.
        Store the model for subsequent access.
    """

    def __init__(self, userConfig : UserConfig, day):
        """
            constructor

            user_dir: path to the user's directory, files are already processed per day
            model_dir: path to the user's model file
            day: day number to be used for training, eval will be day+1

        """
        multiprocessing.Process.__init__(self)
        self.userConfig = userConfig
        self.day = day
        self.logger = logging.getLogger('ProcessUser: {}'.format(self.userConfig.user_name))
        self.logger.setLevel(logging.INFO)
        
    def run(self):

        
        time.sleep(1)
        # file_name = "{}/{}.txt".format(self.userConfig.feat_dir, self.day)
        # logger.info("Processing file:", file_name)

        # load model if exist or create a new one
        char_lm = lm.KerasLM(self.userConfig)

        # train model
        self.logger.info("%s Training step...", self.userConfig.user_name)
        char_lm.do_training(self.day)

        self.logger.info("%s Evaluation step...", self.userConfig.user_name)
        next_day = int(self.day)+1        
        char_lm.do_evaluate(next_day)


# user_names = ['U12', 'U13', 'U24', 'U78', 'U207', 'U293', 'U453', 'U679', 'U1289', 'U1480']
user_names = ['U207', 'U293']
users_indir = '../data/users_feats'
users_lossdir = '../data/users_losses'
users_modeldir = '../data/users_models'
users_logidr = '../data/users_logs'

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

    for d in range(27): # three days
        process_list = []
        for u in user_names:
            logger.info('Processing files for User: %s', u)
            userConfig = UserConfig()
            userConfig.user_name = u
            userConfig.feat_dir = '{0}/{1}/'.format(users_indir, u)
            userConfig.output_filepath = '{0}/{1}_losses.txt'.format(users_lossdir, u)
            userConfig.model_filepath = '{0}/{1}_simple_lm.hdf5'.format(users_modeldir, u)
            userConfig.log_filepath = '{}/{}_log.txt'.format(users_logidr, u)

            process_user = ProcessUser(userConfig, d)
            process_list.append(process_user)

        logger.info(".... processing day: %d", d)
        for p in process_list:
            p.start()

        logger.info("... waiting for processing to finish for day: %d", d)
        for p in process_list:
            p.join()
        
        tf.keras.backend.clear_session()

