import os
import threading
import numpy as np

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import char_simple_lm as lm

tf.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))


class ProcessUser(object):
    """
        Process user log for one day training the model then evaluate the day after
        In a separate thread. 
        Load the model and train it if available.
        Store the model for subsequent access.
    """

    def __init__(self, user_datadir, model_filepath, output_filepath, day):
        """
            constructor

            user_dir: path to the user's directory, files are already processed per day
            model_dir: path to the user's model file
            day: day number to be used for training, eval will be day+1

        """
        self.user_datadir = user_datadir
        self.model_filepath = model_filepath
        self.output_filepath = output_filepath
        self.day = day

        thread = threading.Thread(target=self.run(), args=())
        # thread.daemon = True # daemonized thread
        thread.start()

    def run(self):
        file_name = "{}/{}.txt".format(self.user_datadir, self.day)
        print("Processing file:", file_name)

        # load model if exist or create a new one
        char_lm = lm.SimpleLM(self.user_datadir, self.model_filepath)

        # train model
        print("Training model...")
        char_lm.do_training(self.day)

        next_day = int(self.day)+1
        eval_file_name = "{}/{}.txt".format(self.user_datadir, next_day)
        print("Evaluating file:", eval_file_name)
        char_lm.do_evaluate(next_day, self.output_filepath)


user_names = ['U12', 'U13', 'U24', 'U78', 'U207', 'U293', 'U453', 'U679', 'U1289', 'U1480']
users_indir = '../data/users_feats'
users_lossdir = '../data/users_loss'
users_modeldir = '../data/users_model'


if __name__ == "__main__":

    if not os.path.exists(users_lossdir):
        os.makedirs(users_lossdir)

    if not os.path.exists(users_modeldir):
        os.makedirs(users_modeldir)
    
    for d in range(1): # three days
        for u in user_names:
            print('Processing files for User:', u)
            user_dir = '{0}/{1}/'.format(users_indir, u)
            outfile_name = "{0}/{1}_losses.txt".format(users_lossdir, u)
            model_filepath = '{0}/{1}_simple_lm.hdfs'.format(users_modeldir, u)
            
            process_user = ProcessUser(user_dir, model_filepath, outfile_name, d)
