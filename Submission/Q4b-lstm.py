
# coding: utf-8
import sys
import time
import numpy as np
import os

from utils import *
from rnnmath import *
from sys import stdout
from rnn import *

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, GRU, Dropout
from keras.models import Model
import pandas as pd
#import matplotlib.pyplot as plt

data_folder = "../data"


# In[55]:


vocab_size = 2000
vocab = pd.read_table(data_folder + "/vocab.wiki.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
num_to_word = dict(enumerate(vocab.index[:vocab_size]))
word_to_num = invert_dict(num_to_word)

# calculate loss vocabulary words due to vocab_size
fraction_lost = fraq_loss(vocab, word_to_num, vocab_size)
print("Retained %d words from %d (%.02f%% of all tokens)\n" % (vocab_size, len(vocab), 100*(1-fraction_lost)))

# load training data
sents = load_np_dataset(data_folder + '/wiki-train.txt')
S_train = docs_to_indices(sents, word_to_num, 0, 0)
X_train, D_train = seqs_to_npXY(S_train)

sents = load_np_dataset(data_folder + '/wiki-dev.txt')
S_dev = docs_to_indices(sents, word_to_num, 0, 0)
X_dev, D_dev = seqs_to_npXY(S_dev)

sents = load_np_dataset(data_folder + '/wiki-test.txt')
S_test = docs_to_indices(sents, word_to_num, 0, 0)
X_test, D_test = seqs_to_npXY(S_test)


# X_len = list(map(lambda x: len(x), X_train))
# pd.Series(X_len).hist()
# plt.show()

EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 20
x_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH, dtype='int32', padding='post', truncating='post', value = word_to_num["<s>"])
x_dev = pad_sequences(X_dev, maxlen=MAX_SEQUENCE_LENGTH, dtype='int32', padding='post', truncating='post', value = word_to_num["<s>"])
x_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH, dtype='int32', padding='post', truncating='post', value = word_to_num["<s>"])


embeddings_index = {}
with open(os.path.join(data_folder, 'glove.6B/glove.6B.100d.txt'), "r") as f:
    lines = f.read().split("\n")
    for line in lines:
        values = line.split()
        if len(values) > 0:
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs



embedding_unk = embeddings_index.get("unk")
embedding_matrix = np.zeros((len(word_to_num) + 1, EMBEDDING_DIM))
for word, i in word_to_num.items():
    embedding_vector = embeddings_index.get(word, embedding_unk)
    embedding_matrix[i] = embedding_vector



embedding_layer = Embedding(len(num_to_word) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)



sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
lstm = LSTM(64, input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)) (embedded_sequences)
dense_ = Dense(200, activation="relu")(lstm)
out = Dense(1, activation="sigmoid")(dense_)


model = Model(sequence_input, out)
callbacks = [
    EarlyStopping(monitor='val_acc', patience=5, verbose=1),
    ModelCheckpoint("weights.h5", monitor='val_loss', save_best_only=True, verbose=1)]

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
model.summary()
model.fit(x_train, D_train, validation_data=(x_dev, D_dev),
          epochs=50, batch_size=100, callbacks = callbacks)


#at the end

model.load_weights("weights.h5")
preds = model.predict(x_test)
acc = np.mean((preds > 0.5) == D_test.reshape(-1, 1))
print("Test accuracy: ", acc)

from keras.utils import plot_model
plot_model(model, to_file='model2.png', show_shapes=True)
