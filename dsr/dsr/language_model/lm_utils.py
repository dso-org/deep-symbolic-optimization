"""
- token library
- preprocess for training, test predicting, dsr
"""


import numpy as np
from sklearn.model_selection import train_test_split

import pickle

SEED = 2020
MAX = 31

data_token_library={
    '<end>':0,
    'TERMINAL':1, #symbol, integer
    'add':2,
    'sub':3,
    'mul':4,
    'div':5,
    'sin':6,
    'cos':7,
    'exp':8,
    'log':9
}

def build_dataset_no_eos(filename=None, predict_input=None, tokens=None, test_size=0.1):
    # [3,1,1,0,0,...,3]: [data] + [length of data] for dynrnn

    if predict_input is None:
        with open(filename,'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = predict_input

    if tokens is None:
        # tokens = ['TERMINAL','add','sub','mul','div','sin','cos','exp','log','<end>']
        tokens=data_token_library.keys()
    # Creating a mapping from unique characters to indices
    token2idx = {u:i for i, u in enumerate(tokens)}
    #~ idx2token = np.array(tokens)

    dataset_use = []
    # remove <end>
    for d in dataset:
        if d[-1] == token2idx['<end>']:
            d = d[:-1]
        # d.insert(0,token2idx['<end>'])
        d = np.array(d)
        seq_len = len(d)-1

        if len(d) > MAX: continue
        d = np.pad(d,(0,MAX-len(d)),'constant',constant_values=token2idx['<end>']) # 
        d = np.append(d,seq_len)
        dataset_use.append(np.array(d))
    dataset_use=np.array(dataset_use)

    if predict_input is None:
        data_train, data_test = train_test_split(dataset_use, test_size=test_size, random_state=SEED, shuffle=True)
        return data_train, data_test, token2idx
    else:
        return dataset_use, dataset_use, token2idx


def build_dataset_with_eos_padding(filename=None, predict_input=None, tokens=None, test_size=0.1):

    if predict_input is None:
        with open(filename,'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = predict_input

    if tokens is None:
        # tokens = ['TERMINAL','add','sub','mul','div','sin','cos','exp','log','<end>']
        tokens=data_token_library.keys()
    # Creating a mapping from unique characters to indices
    token2idx = {u:i for i, u in enumerate(tokens)}
    #~ idx2token = np.array(tokens)

    # MAX sequence length
    # MAX = max([len(d) for d in dataset])

    dataset_use = []
    # # ZERO PADDING
    for d in dataset:
        d = np.array(d)
        if len(d) > MAX: continue
        d = np.pad(d,(0,MAX-len(d)),'constant',constant_values=token2idx['<end>']) # padding for constant length
        dataset_use.append(d)
    dataset_use=np.array(dataset_use)

    if predict_input is None:
        data_train, data_test = train_test_split(dataset_use, test_size=test_size, random_state=SEED, shuffle=True)
        return data_train, data_test, token2idx
    else:
        return dataset_use, dataset_use, token2idx


def batch_iter(inputs, batch_size, num_epochs):
    inputs = np.array(inputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for _ in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index]