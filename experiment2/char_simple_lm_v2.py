import os
import time
import numpy as np

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))


user_names = ['U12', 'U13', 'U24', 'U78', 'U207', 'U293', 'U453', 'U679', 'U1289', 'U1480']
users_indir = '../data/users_feats'
users_lossdir = '../data/users_loss'
users_modeldir = '../data/users_model'

max_len = 120 # max length of sentence
num_chars = 128 # our vocabulary, i.e. unique characters in text. We'll just use the first 128 (half ASCII)


# transform character-based input into equivalent numerical versions
def encode_data(text, num_chars, max_length):
    # create empty vessels for one-hot encoded input
    X = np.zeros((len(text), max_length, num_chars), dtype=np.float32)
    y = np.zeros((len(text), max_length, num_chars), dtype=np.float32)
    
    # loop over inputs and tranform and store in X
    for i, sentence in enumerate(text):
        sentence = '\t' + sentence + '\n'
        for j, c in enumerate(sentence):
            X[i, j, ord(c)] = 1
            if j > 0:
                # target_data will be ahead by one timestep
                # and will not include the start character.
                y[i, j - 1, ord(c)] = 1.

    return X, y


def process_file(fname):
    """
        process file by extracting sentences data and encode them producing 
        a set of input and target data for processing by the model
        'fname' contains coma separated fields where the last one is the 
        sentence to be processes
    """
    data = open(fname).read()

    text = []
    red_events = []
    with open(fname, 'r') as infile:
        for i, line in enumerate(infile.readlines()):
            line = line.strip().split(',')
            text.append(line[-1])
            if int(line[2]) == 1:
                red_events.append((i,line))
        infile.close()
#     print(text[0], 'len:', len(text[0]), len(text))

    input_data, target_data = encode_data(text, num_chars, max_len)
    
    return input_data, target_data, red_events
    
    
def getModel():
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(20, input_shape=(None, num_chars), return_sequences=True),  # input shape required
#         tf.keras.layers.Dense(num_chars, activation="relu"),
        tf.keras.layers.Dense(num_chars, activation="softmax"),
    ])
    return model

def loss(model, x, y):
    y_ = model(x)
    return tf.keras.losses.categorical_crossentropy(y, y_)

def grad(model, inputs, targets):
    with tfe.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)


# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, epsilon=1e-08)


def train(model, training_dataset, num_epochs):
    
    train_losses = []
    loss_results = []
    
    for epoch in range(num_epochs):
        epoch_loss_avg = tfe.metrics.Mean()

        startTime = time.time()
        # training using batches of 'batch_size'
        for X, y in tfe.Iterator(training_dataset):
            grads = grad(model, X, y)
            optimizer.apply_gradients(zip(grads, model.variables), 
                                     global_step=tf.train.get_or_create_global_step())
            batch_loss = loss(model, X, y)
            epoch_loss_avg(batch_loss) # batch loss
            train_losses.append(tf.reduce_mean(batch_loss))

        loss_results.append(epoch_loss_avg.result())

        if epoch % 1 == 0:
            print("Epoch {:03d}: Loss: {:.3f} - in: {:.3f} sec.".format(epoch, 
                                                                        epoch_loss_avg.result(), 
                                                                        (time.time()-startTime)))        
#         del epoch_loss_avg
        
    avg_loss = tf.reduce_mean(train_losses)
    max_loss = tf.reduce_max(train_losses)
    print('  avg_loss:', avg_loss, ' - max_loss:', max_loss)
 
    return loss_results


def process_user(user_name, outfile):
    num_epochs = 2
    batch_size = 512
    data_dir = '{0}/{1}/'.format(users_indir, u)
    
    # keep results for plotting
    train_loss_results = []

    model = getModel()

    for d in range(4):

        # training phase
        dataset_fname = data_dir+'{0}.txt'.format(d)
#         print('df:', dataset_fname)
        input_data, target_data, red_events = process_file(dataset_fname)
        print('processing:', dataset_fname, " - num events:", len(input_data), " - red events:", len(red_events))

        training_dataset = tf.data.Dataset.from_tensor_slices((input_data, target_data))
        training_dataset = training_dataset.batch(batch_size)

        # train model on a day
        loss_results = train(model, training_dataset, num_epochs)
        train_loss_results.append(loss_results)
        print('loss_results:', loss_results)
        
        # some cleanup
        input_data = None
        target_data = None
        training_dataset = None
        
        # evaluation phase
        dataset_fname = data_dir+'{0}.txt'.format(d+1)
        input_data, target_data, red_events = process_file(dataset_fname)
        print('  evaluating:', dataset_fname, " - num events:", len(input_data), " - red events:", len(red_events))

        eval_dataset = tf.data.Dataset.from_tensor_slices((input_data, target_data))
        eval_dataset = eval_dataset.batch(batch_size)

        line_losses = np.array([])
        
        # eval using batches of 'batch_size'
        for X, y in tfe.Iterator(eval_dataset):
            line_losses = np.append(line_losses, tf.reduce_mean(loss(model, X, y), axis=1))

        possible_anomalies = [(i,v) for i, v in enumerate(line_losses)]
        possible_anomalies.sort(key=lambda x: x[1], reverse=True)

        # some cleanup
        input_data = None
        target_data = None
        eval_dataset = None
        line_losses = None

        print('    max:', possible_anomalies[:10])
        print('    red events:', [a for a,b in red_events])
        
        # write top 10 losses to a file with the format (day, score, redevent)
        for i, v in possible_anomalies[:20]:
            red = '0'
            for a,b in red_events:
                if a == i:
                    red = '1'
                    break
            line = '{0},{1},{2}\n'.format(d, v, red)
            outfile.write(line)
      
    # Save model to a file
    model_filepath = '{0}/{1}_simple_lm.hdfs'.format(users_modeldir, user_name)

    tf.keras.models.save_model(
        model,
        model_filepath,
        overwrite=True,
        include_optimizer=False
    )

    model = None
    

if __name__ == "__main__":
    if not os.path.exists(users_lossdir):
        os.makedirs(users_lossdir)

    if not os.path.exists(users_modeldir):
        os.makedirs(users_modeldir)
    
    for u in user_names:
        print('Processing files for User:', u)
        outfile_name = "{0}/{1}_losses.txt".format(users_lossdir, u)

        with open(outfile_name, 'w') as outfile:
            outfile.write('day,loss,redevent\n')
            if tfe.num_gpus() > 0:
                with tf.device("/gpu:0"):
                    process_user(u, outfile)
            else: # run on CPU
                process_user(u, outfile)
            outfile.close()
