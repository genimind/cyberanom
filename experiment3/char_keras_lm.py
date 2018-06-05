import os
import time
import logging

import numpy as np
from sklearn.metrics import mean_squared_error, log_loss

import tensorflow as tf
# import tensorflow.contrib.eager as tfe

from process_utils import process_file, UserConfig

max_len = 120 # max length of sentence
num_chars = 128 # our vocabulary, i.e. unique characters in text. We'll just use the first 128 (half ASCII)

log_level = logging.INFO
verbosity = 0
debug = False

if debug:
    log_level = logging.DEBUG
    verbosity = 1

def getBidiModel():
    model = tf.keras.Sequential([
        tf.keras.layers.Masking(mask_value=0., input_shape=(None, num_chars)),
        # tf.keras.layers.InputLayer(input_shape = (None, num_chars)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(40, return_sequences=True)),  # input shape required\n",
    #         tf.keras.layers.Dense(240, activation="relu"),
        # tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_chars, activation="softmax")
    ])
    return model


class KerasLM(object):
    """
        simple bidirectional keras LSTM model with train and evaluate interfaces
    """

    def __init__(self, userConfig : UserConfig):
        """
        constructor.
        
        model_path: load model from path if exist, path is also used to store the model after training

        """
                
        self.userConfig = userConfig
        self.batch_size = 256
        self.logger = logging.getLogger('{}_({})'.format(__name__, self.userConfig.user_name))
        handler = logging.FileHandler(self.userConfig.log_filepath)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(log_level)

        self.csv_logger = tf.keras.callbacks.CSVLogger(self.userConfig.log_filepath, append=True)
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0005, patience=1, verbose=2, mode='min')

        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        self.optimizer = tf.keras.optimizers.RMSprop(lr=0.001, epsilon=1e-08)

        self.logger.info('model_filepath: %s', self.userConfig.model_filepath)
        if os.path.exists(self.userConfig.model_filepath):
            self.logger.info('loading model:')
            self.model = tf.keras.models.load_model(self.userConfig.model_filepath,
                                        custom_objects=None,
                                        compile=True)
        else:
            self.logger.info('creating new model')
            self.model = getBidiModel()

            self.logger.info('compiling model')
            self.model.compile(
                loss='categorical_crossentropy',
                optimizer=self.optimizer)
                # metrics=['mean_squared_error'])

        # self.logger.info('Model summary:')
        # self.model.summary()
        # self.logger.info('model metrics names:', self.model.metrics_names)

    def _eval_loss_v2(self, X, y):
        """
            This evaluation loss function uses the same loss function used while training the 
            model. There is an issue with the performance, perhaps due to conversion to tensors and back
        """
        line_losses = np.array([])
        
        y_ = self.model.predict(X, batch_size=self.batch_size, verbose=verbosity)
        self.logger.debug('y shape: %s', y.shape)
        self.logger.debug('y_ shape: %s', y_.shape)

        loss_res = tf.keras.losses.categorical_crossentropy(tf.convert_to_tensor(y), tf.convert_to_tensor(y_))
        
        line_losses = loss_res.eval(session=tf.Session())
        
        line_losses = np.mean(line_losses, axis=1)

        return line_losses


    def _eval_loss(self, X, y):
        """
            This version of the evaluation loss function uses the same type of loss function from sklearn
        """
        y_ = self.model.predict(X, batch_size=self.batch_size, verbose=verbosity)
        self.logger.debug('y shape: %s', y.shape)
        self.logger.debug('y_ shape: %s', y_.shape)

        line_losses = []
        for l in range(y.shape[0]):
            line_losses.append(log_loss(y[l], y_[l]))

        return line_losses
        

    def _train(self, X, y, num_epochs):
        
        self.logger.debug('X shape: %s  - y shape: %s', X.shape, y.shape)

        history = self.model.fit(X, y, batch_size=self.batch_size, epochs=num_epochs, verbose=verbosity,
                                     callbacks=[self.csv_logger, self.early_stopping])
#        history = self.model.fit(X, y, batch_size=self.batch_size, epochs=num_epochs, verbose=2)
        avg_loss = np.mean(history.history['loss'])
        return avg_loss

    def do_training(self, day):
        num_epochs = 10

        dataset_fname = self.userConfig.feat_dir+'{}.txt'.format(day)
        
        # check if file exist
        if not os.path.exists(dataset_fname):
            return False

        input_data, target_data, red_events = process_file(dataset_fname, num_chars, max_len)
        self.logger.info('processing: %s - num events: %d  - red events:%d', dataset_fname, len(input_data), len(red_events))
        
        if len(input_data) == 0: # nothing in the file
            return False

        # train model on a day
        loss_results = self._train(input_data, target_data, num_epochs)

        self.logger.info('training avg loss: %s', loss_results)

        # Save model to a file
        self.model.save(self.userConfig.model_filepath)

        return True

    def do_evaluate(self, day):
        """     
        Evaluation phase
        """
        dataset_fname = self.userConfig.feat_dir+'{}.txt'.format(day)
        # check if file exist
        if not os.path.exists(dataset_fname):
            return False

        input_data, target_data, red_events = process_file(dataset_fname, num_chars, max_len)
        self.logger.info('  evaluating: %s - num events: %d  - red events:%d', dataset_fname, len(input_data), len(red_events))

        line_losses = self._eval_loss(input_data, target_data)

        self.process_anomalies_for_max(day, line_losses, red_events)

        avg_loss = np.average(line_losses)
        line_losses_diff = line_losses - avg_loss
        self.logger.info('  avg eval loss: %s', avg_loss)

        self.process_anomalies_for_diff(day, line_losses_diff, red_events)

        return True


    def process_anomalies_for_max(self, day, line_losses, red_events):
        """
            just process the values based on the maximum anomalies for the day
            and store information in a file xxx__max.txt
        """
        num_events = len(line_losses)

        possible_anomalies = [(i,v) for i, v in enumerate(line_losses)]
        possible_anomalies.sort(key=lambda x: x[1], reverse=True)

        self.logger.debug('    top max: %s', possible_anomalies[:10])
        self.logger.debug('    red events: %s', [a for a,b in red_events])
        for index, (i, v) in enumerate(possible_anomalies):
            for a, b in red_events:
                if a == i:
                    self.logger.info('      red_event index: %d  - anomaly index: %d', a, index)
        
        # write top 10% losses to a file with the format (day, score, redevent)
        num_to_store = int(num_events * 0.1)
        output_filepath = '{}_max.txt'.format(self.userConfig.output_base_filepath)
        with open(output_filepath, 'a+') as outfile:
            for i, v in possible_anomalies[:num_to_store]:
                red = '0'
                for a,b in red_events:
                    if a == i:
                        red = '1'
                        break
                line = '{},{},{},{}\n'.format(day, num_events, v, red)
                outfile.write(line)
            outfile.close()


    def process_anomalies_for_diff(self, day, line_losses_diff, red_events):
        """
            just process the values using a simple normalization by measuring the diff
            to the average and list the top anomalies for the day which are stored 
            in a file xxx__diff.txt
        """
        num_events = len(line_losses_diff)

        possible_anomalies = [(i,v) for i, v in enumerate(line_losses_diff)]
        possible_anomalies.sort(key=lambda x: x[1], reverse=True)

        self.logger.debug('    top diff: %s', possible_anomalies[:10])
        self.logger.debug('    red events: %s', [a for a,b in red_events])
        for index, (i, v) in enumerate(possible_anomalies):
            for a, b in red_events:
                if a == i:
                    self.logger.info('      red_event index: %d  - anomaly index: %d', a, index)
        
        # write top 100 losses to a file with the format (day, score, redevent)
        num_to_store = int(num_events * 0.1)
        output_filepath = '{}_diff.txt'.format(self.userConfig.output_base_filepath)
        with open(output_filepath, 'a+') as outfile:
            for i, v in possible_anomalies[:num_to_store]:
                red = '0'
                for a,b in red_events:
                    if a == i:
                        red = '1'
                        break
                line = '{},{},{},{}\n'.format(day, num_events, v, red)
                outfile.write(line)
            outfile.close()
