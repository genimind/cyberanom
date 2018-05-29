import os
import time

import numpy as np
from sklearn.metrics import mean_squared_error, log_loss

import tensorflow as tf
# import tensorflow.contrib.eager as tfe

from process_utils import process_file

max_len = 120 # max length of sentence
num_chars = 128 # our vocabulary, i.e. unique characters in text. We'll just use the first 128 (half ASCII)

def getBidiModel():
    model = tf.keras.Sequential([
        tf.keras.layers.Masking(mask_value=0., input_shape=(None, num_chars)),
        # tf.keras.layers.InputLayer(input_shape = (None, num_chars)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),  # input shape required\n",
    #         tf.keras.layers.Dense(240, activation="relu"),
        # tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_chars, activation="softmax")
    ])
    return model


class KerasLM(object):
    """
        simple bidirectional keras LSTM model with train and evaluate interfaces
    """

    def __init__(self, user_datadir, model_filepath):
        """
        constructor.
        
        model_path: load model from path if exist, path is also used to store the model after training

        """
                
        self.model_filepath = model_filepath
        self.user_datadir = user_datadir
        self.batch_size = 256

        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        self.optimizer = tf.keras.optimizers.RMSprop(lr=0.001, epsilon=1e-08)

        print('model_filepath:', model_filepath)
        if os.path.exists(self.model_filepath):
            print('loading model:')
            self.model = tf.keras.models.load_model(self.model_filepath,
                                        custom_objects=None,
                                        compile=True)
        else:
            print('creating new model')
            self.model = getBidiModel()

            print('compiling model')
            self.model.compile(
                loss='categorical_crossentropy',
                optimizer=self.optimizer)
                # metrics=['mean_squared_error'])

        # print('Model summary:')
        # self.model.summary()
        # print('model metrics names:', self.model.metrics_names)

    def eval_loss_v2(self, X, y):
        line_losses = np.array([])
        
        y_ = self.model.predict(X, batch_size=self.batch_size, verbose=1)
        print('y shape:', y.shape)
        print('y_ shape:', y_.shape)

        loss_res = tf.keras.losses.categorical_crossentropy(tf.convert_to_tensor(y), tf.convert_to_tensor(y_))
        
        line_losses = loss_res.eval(session=tf.Session())
        
        line_losses = np.mean(line_losses, axis=1)

        return line_losses

    def eval_loss(self, X, y):

        y_ = self.model.predict(X, batch_size=self.batch_size, verbose=1)
        print('y shape:', y.shape)
        print('y_ shape:', y_.shape)

        line_losses = []
        for l in range(y.shape[0]):
            line_losses.append(log_loss(y[l], y_[l]))

        return line_losses
        

    def train(self, X, y, num_epochs):
        
        train_losses = []
        loss_results = []

        print('X shape:', X.shape, ' - y shape:', y.shape)

        history = self.model.fit(X, y, batch_size=self.batch_size, epochs=num_epochs)
        avg_loss = np.mean(history.history['loss'])
        return avg_loss

    def do_training(self, day):
        num_epochs = 10

        # keep results for plotting
        train_loss_results = []

        # training phase
        dataset_fname = self.user_datadir+'{0}.txt'.format(day)
#         print('df:', dataset_fname)
        input_data, target_data, red_events = process_file(dataset_fname, num_chars, max_len)
        print('processing:', dataset_fname, " - num events:", len(input_data), " - red events:", len(red_events))

        # train model on a day
        loss_results = self.train(input_data, target_data, num_epochs)
        train_loss_results.append(loss_results)
        print('loss_results:', loss_results)

        # Save model to a file

        self.model.save(self.model_filepath)


    def do_evaluate(self, day, output_filepath):
        """     
        Evaluation phase
        """
        dataset_fname = self.user_datadir+'{0}.txt'.format(day)
        input_data, target_data, red_events = process_file(dataset_fname, num_chars, max_len)
        print('  evaluating:', dataset_fname, " - num events:", len(input_data), " - red events:", len(red_events))

        line_losses = self.eval_loss(input_data, target_data)

        possible_anomalies = [(i,v) for i, v in enumerate(line_losses)]
        possible_anomalies.sort(key=lambda x: x[1], reverse=True)

        print('    max:', possible_anomalies[:10])
        print('    red events:', [a for a,b in red_events])
        for index, (i, v) in enumerate(possible_anomalies):
            for a, b in red_events:
                if a == i:
                    print('      red_event index:', a, ' - anomaly index:', index)
        
        # write top 20 losses to a file with the format (day, score, redevent)
        with open(output_filepath, 'w+') as outfile:
            for i, v in possible_anomalies[:20]:
                red = '0'
                for a,b in red_events:
                    if a == i:
                        red = '1'
                        break
                line = '{0},{1},{2}\n'.format(day, v, red)
                outfile.write(line)
            outfile.close()