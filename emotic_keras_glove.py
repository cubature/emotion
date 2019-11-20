'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classification of newsgroup messages into 20 different categories).
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)
20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

from __future__ import print_function

import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.initializers import Constant
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, LSTM, MaxPooling1D
from tensorflow.keras.models import Sequential
import pickle


BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, './glove')
#TEXT_DATA_DIR = os.path.join(BASE_DIR, './data_glove/')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 10000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
path_train_txt = './data/train.txt'
path_test_txt = './data/test.txt'

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs


# fonction pour lire les fichiers textes
def read_txt_file(path):
    x = []
    y = []
    with open(path, 'r', encoding='utf-8') as input_file:
        for line in input_file.readlines():
            [sentence, emotion] = line.split(';')
            if len(sentence) == 0:
                continue
            x.append(sentence)
            y.append(emotion)
    return x, y


x_train, y_train = read_txt_file(path_train_txt)
x_test, y_test = read_txt_file(path_test_txt)

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
x_train_encoded = pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=MAX_SEQUENCE_LENGTH)
x_test_encoded = pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=MAX_SEQUENCE_LENGTH)
y_train_encoded = to_categorical(y_train, 5)
y_test_encoded = to_categorical(y_test, 5)

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(MAX_NUM_WORDS,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                           trainable=False)

# saving the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Training model.')
model = Sequential()
model.add(Embedding(10001, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_matrix], trainable=False))
# model.add(Conv1D(8, 5, activation='relu'))
# model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(10, activation='relu', return_sequences=True))
model.add(LSTM(10))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['categorical_accuracy'])
history = model.fit(x_train_encoded, y_train_encoded,
                    batch_size=512,
                    epochs=1,
                    verbose=1, validation_split=0.1)

# Save the weights
model.save_weights('model_weights.h5')

# Save the model architecture
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())

score = model.evaluate(x_test_encoded, y_test_encoded, batch_size=128, verbose=1)

print('train accuracy : ', history.history['categorical_accuracy'])
print('val accuracy : ', history.history['val_categorical_accuracy'])
print('Test accuracy : ', score)
