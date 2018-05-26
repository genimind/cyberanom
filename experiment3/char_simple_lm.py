import os
import time

import numpy as np

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from process_utils import process_file

max_len = 120 # max length of sentence
num_chars = 128 # our vocabulary, i.e. unique characters in text. We'll just use the first 128 (half ASCII)

def getModel():
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(20, input_shape=(None, num_chars), return_sequences=True),  # input shape required
#         tf.keras.layers.Dense(num_chars, activation="relu"),
        tf.keras.layers.Dense(num_chars, activation="softmax"),
    ])
    return model



class SimpleLM(object):
    """
        simple LSTM model with train and evaluate interfaces
    """

    def __init__(self, user_datadir, model_filepath):
        """
        constructor.
        
        model_path: load model from path if exist, path is also used to store the model after training

        """
        self.model_filepath = model_filepath
        self.user_datadir = user_datadir
        self.batch_size = 512

        print('model_filepath:', model_filepath)
        if os.path.exists(self.model_filepath):
            self.model = tf.keras.models.load_model(self.model_filepath,
                                        custom_objects=None,
                                        compile=False)
        else:
            self.model = getModel()

        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, epsilon=1e-08)


    def loss(self, x, y, isTraining = None):
        y_ = self.model(x, training = isTraining)
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, y_))


    def grad(self, inputs, targets, isTraining):
        with tfe.GradientTape() as tape:
            loss_value = self.loss(inputs, targets, isTraining)
        return tape.gradient(loss_value, self.model.variables), loss_value




    def train(self, training_dataset, num_epochs):
        
        train_losses = []
        loss_results = []
        
        for epoch in range(num_epochs):
            epoch_loss_avg = tfe.metrics.Mean()

            startTime = time.time()
            # training using batches of 'batch_size'
            for X, y in tfe.Iterator(training_dataset):
                
                grads, batch_loss = self.grad(X, y, isTraining=True)
                self.optimizer.apply_gradients(zip(grads, self.model.variables), 
                                        global_step=tf.train.get_or_create_global_step())
                epoch_loss_avg(batch_loss) # batch loss
                train_losses.append(batch_loss)

            loss_results.append(epoch_loss_avg.result())

            if epoch % 1 == 0:
                print("Epoch {:03d}: Loss: {:.3f} - in: {:.3f} sec.".format(epoch, 
                                                                            epoch_loss_avg.result(), 
                                                                            (time.time()-startTime)))        
            
        avg_loss = tf.reduce_mean(train_losses)
        max_loss = tf.reduce_max(train_losses)
        print('  avg_loss:', avg_loss, ' - max_loss:', max_loss)
    
        return loss_results


    def do_training(self, day):
        num_epochs = 2

        # keep results for plotting
        train_loss_results = []

        # training phase
        dataset_fname = self.user_datadir+'{0}.txt'.format(day)
#         print('df:', dataset_fname)
        input_data, target_data, red_events = process_file(dataset_fname, num_chars, max_len)
        print('processing:', dataset_fname, " - num events:", len(input_data), " - red events:", len(red_events))

        training_dataset = tf.data.Dataset.from_tensor_slices((input_data, target_data))
        training_dataset = training_dataset.batch(self.batch_size)

        # train model on a day
        loss_results = self.train(training_dataset, num_epochs)
        train_loss_results.append(loss_results)
        print('loss_results:', loss_results)

        # Save model to a file

        tf.keras.models.save_model(
            self.model,
            self.model_filepath,
            overwrite=True,
            include_optimizer=False
        )


    def do_evaluate(self, day, output_filepath):
        """     
        Evaluation phase
        """
        dataset_fname = self.user_datadir+'{0}.txt'.format(day)
        input_data, target_data, red_events = process_file(dataset_fname, num_chars, max_len)
        print('  evaluating:', dataset_fname, " - num events:", len(input_data), " - red events:", len(red_events))

        eval_dataset = tf.data.Dataset.from_tensor_slices((input_data, target_data))
        eval_dataset = eval_dataset.batch(self.batch_size)

        line_losses = np.array([])
        
        # eval using batches of 'batch_size'
        for X, y in tfe.Iterator(eval_dataset):
            batch_loss = self.loss(X, y)
            line_losses = np.append(line_losses, batch_loss)

        possible_anomalies = [(i,v) for i, v in enumerate(line_losses)]
        possible_anomalies.sort(key=lambda x: x[1], reverse=True)

        print('    max:', possible_anomalies[:10])
        print('    red events:', [a for a,b in red_events])
        
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