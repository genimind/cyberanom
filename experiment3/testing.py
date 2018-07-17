import os
import time
import multiprocessing
import numpy as np
import logging
import tensorflow as tf
import char_keras_lm as lm
from process_utils import process_file, UserConfig


if __name__ == "__main__":

    users_indir = '../data/users_feats'
    users_lossdir = '../data/test/users_losses'
    users_modeldir = '../data/exper3__all__1epoch__bidi_model/users_models'
    users_logidr = '../data/test/users_logs'

    u = 'U12'
    userConfig = UserConfig()
    userConfig.user_name = u
    userConfig.feat_dir = '{0}/{1}/'.format(users_indir, u)
    userConfig.output_base_filepath = '{0}/{1}_losses'.format(users_lossdir, u)
    userConfig.model_filepath = '{0}/{1}_simple_lm.hdf5'.format(users_modeldir, u)
    userConfig.log_filepath = '{}/{}_log.txt'.format(users_logidr, u)


    day = 10
    char_lm = lm.KerasLM(userConfig)
    dataset_fname = userConfig.feat_dir+'{}.txt'.format(day)

    input_data, target_data, red_events = process_file(dataset_fname, num_chars, max_len)
    self.logger.info('  evaluating: %s - num events: %d  - red events:%d', dataset_fname, len(input_data), len(red_events))

    ### see testing_model.ipynb for implementation
