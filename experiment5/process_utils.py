import numpy as np

class UserConfig :
    def __init__(self):
        self.user_name = 'unknown'
        self.feat_dir = 'unknown'
        self.output_base_filepath = 'unknown'
        self.model_filepath = 'unknwon'
        self.log_filepath = 'unknown'


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


def decode_data(data):
    results=[]
    # print('data.shape:', data.shape)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                if data[i, j, k] == 1:
                    results.append(chr(k))
                    break
    
    return ''.join(results)


def process_file(fname, num_chars, max_len, filter_red=False):
    """
        process file by extracting sentences data and encode them producing 
        a set of input and target data for processing by the model
        'fname' contains coma separated fields where the last one is the 
        sentence to be processes

        fname: file name
        num_chars: num of ASCII used for encoding
        max_len: max encoding array
        filter_red: if True don't add red events to the encoded text (used when training)
    """
    
    text = []
    red_events = []
    with open(fname, 'r') as infile:
        for i, line in enumerate(infile.readlines()):
            red_event = False
            line = line.strip().split(',')
            if int(line[2]) == 1:
                red_events.append((i,line))
                red_event = True
            if filter_red  and red_event:
                continue
            else:
                data = line[-1].split('|')
                text.append(','.join(data[0:]))
        infile.close()
    print(text[0], 'len:', len(text[0]), len(text))

    input_data, target_data = encode_data(text, num_chars, max_len)
    
    return input_data, target_data, red_events
