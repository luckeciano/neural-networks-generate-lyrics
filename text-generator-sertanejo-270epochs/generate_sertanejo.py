from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import argparse
import ast
import codecs


def generate_text(model, length, vocab_size, ix_to_char, initial_string):
# starting with random character
	#ix = [np.random.randint(vocab_size)]
	char_to_ix = inv_map = {v: k for k, v in ix_to_char.iteritems()}
	y_char = list(initial_string)
	X = np.zeros((1, length, vocab_size))
	for i in range(len(y_char)):
		print(y_char[i], end = "")
		X[0, i, :][char_to_ix[y_char[i]]] = 1	
	for i in range(len(y_char), length):
		# appending the last predicted character to sequence
		ix = np.argmax(model.predict(X[:, :i, :])[0], 1)
		X[0, i, :][ix[-1]] = 1
		print(ix_to_char[ix[-1]], end="")
		y_char.append(ix_to_char[ix[-1]])
	return ('').join(y_char)


def load_dict (dict_dir):
	s = open(dict_dir, 'r').read()
	print(s)
	mydict = eval(s)
	return   mydict


# Parsing arguments for Network definition
ap = argparse.ArgumentParser()
ap.add_argument('-layer_num', type=int, default=2)
ap.add_argument('-seq_length', type=int, default=50)
ap.add_argument('-hidden_dim', type=int, default=500)
ap.add_argument('-generate_length', type=int, default=500)
ap.add_argument('-weights', default='')
ap.add_argument('-initial', default='')
ap.add_argument('-dictionary', default ='./data/dictionary_funk.txt')
args = vars(ap.parse_args())


HIDDEN_DIM = args['hidden_dim']
SEQ_LENGTH = args['seq_length']
WEIGHTS = args['weights']
INITIAL_STRING = args['initial']
DICTIONARY = args['dictionary']

GENERATE_LENGTH = args['generate_length']
LAYER_NUM = args['layer_num']

# Creating training data
ix_to_char = load_dict(DICTIONARY)
print(ix_to_char)
VOCAB_SIZE = len(ix_to_char)
print(VOCAB_SIZE)

# Creating and compiling the Network
model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
model.add(Dropout(0.3))
for i in range(LAYER_NUM - 1):
  model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

#WEIGHTS = 'checkpoint_layer_3_hidden_700_epoch_10.hdf5'

model.load_weights(WEIGHTS)

result = generate_text(model, args['generate_length'], VOCAB_SIZE, ix_to_char, 'Meu nome')
with open('result.txt', 'w+') as f:
	f.write(result)


