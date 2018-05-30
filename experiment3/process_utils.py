import numpy as np

class UserConfig :
    def __init__(self):
        self.user_name = 'unknown'
        self.feat_dir = 'unknown'
        self.output_filepath = 'unknown'
        self.model_filepath = 'unknwon'


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


def process_file(fname, num_chars, max_len):
    """
        process file by extracting sentences data and encode them producing 
        a set of input and target data for processing by the model
        'fname' contains coma separated fields where the last one is the 
        sentence to be processes
    """
    
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
