import tensorflow as tf
import pickle
import numpy as np
from keras.layers import Dense, Dropout, Activation, SimpleRNN, LSTM, Bidirectional, Conv1D, Flatten, MaxPooling1D# delete "tensorflow." before "keras"
from keras.models import Sequential
from keras.preprocessing import sequence
import tools as tools

# define the documents
path_train_txt = './data/train.txt'
path_test_txt = './data/test.txt'

# paramètres à faire varier pour les tests
input_struct = 'bag_of_words'
texts_to_matrix_mode = 'count'
model_type = 'cnn'
model_loss_function = 'categorical_crossentropy'
model_optimizer = 'Adagrad'
activation_function = 'relu'
num_words = 10000
max_length = 40 # max length (# of words) of each sentences
time_step = max_length
input_dim = num_words
filter_size = 3 # filter size for CNN
embedding_dim = 1 # tokenizer's embedding size is 1
nb_layer = 2
batch_size = 512
nb_neuron = 64
epochs = 2

# constant parameter
number_of_category = 5

# pre-processing of data
x_train, y_train = tools.read_txt_file(path_train_txt)
x_test, y_test = tools.read_txt_file(path_test_txt)
t = tf.keras.preprocessing.text.Tokenizer(num_words=num_words+1) # 0 is always reserved
t.fit_on_texts(x_train)

if input_struct == 'bag_of_words':
	x_train_encoded = t.texts_to_matrix(x_train, mode=texts_to_matrix_mode)[:, 1:] # https://github.com/keras-team/keras/issues/8583
	x_test_encoded = t.texts_to_matrix(x_test, mode=texts_to_matrix_mode)[:, 1:]
	if model_type == 'rnn' or model_type == 'lstm' or model_type == 'bi_lstm' or model_type == 'cnn':
		x_train_encoded = np.reshape(x_train_encoded, (x_train_encoded.shape[0], 1, x_train_encoded.shape[1]))
		x_test_encoded = np.reshape(x_test_encoded, (x_test_encoded.shape[0], 1, x_test_encoded.shape[1]))
		time_step = 1
	if model_type == 'cnn_lstm':
		x_train_encoded = np.reshape(x_train_encoded, (x_train_encoded.shape[0], x_train_encoded.shape[1], 1))
		x_test_encoded = np.reshape(x_test_encoded, (x_test_encoded.shape[0], x_test_encoded.shape[1], 1))
elif input_struct == 'sequence_of_words':
	x_train_encoded = t.texts_to_sequences(x_train)
	x_test_encoded = t.texts_to_sequences(x_test)
	x_train_encoded = sequence.pad_sequences(x_train_encoded, maxlen=max_length, padding='post', truncating='post')
	x_test_encoded = sequence.pad_sequences(x_test_encoded, maxlen=max_length, padding='post', truncating='post')
	x_train_encoded = np.reshape(x_train_encoded, (x_train_encoded.shape[0], x_train_encoded.shape[1], 1))
	x_test_encoded = np.reshape(x_test_encoded, (x_test_encoded.shape[0], x_test_encoded.shape[1], 1))
	input_dim = 1


y_train_encoded = tf.keras.utils.to_categorical(y_train, number_of_category)
y_test_encoded = tf.keras.utils.to_categorical(y_test, number_of_category)

# tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = 4)
# texts = ['a b b c c c', 'a b c']
# tokenizer.fit_on_texts(texts)
# sentences = sequence.pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=4, padding='post', truncating='post')
# print(sentences)
# print(str(sentences.shape[0]))
# exit()

# saving the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
	pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)

# definition of the model
model = Sequential()
if model_type == 'feed_forward':
	for x in range(nb_layer):
		model.add(Dense(nb_neuron, input_shape=(input_dim,)))
		model.add(Activation(activation_function))
		model.add(Dropout(0.5))
elif model_type == 'rnn':
	for x in range(nb_layer - 1):
		model.add(SimpleRNN(units=nb_neuron, activation=activation_function, dropout=0.5, input_shape=(time_step, input_dim), return_sequences=True))
	model.add(SimpleRNN(units=nb_neuron, activation=activation_function, dropout=0.5, input_shape=(time_step, input_dim)))
elif model_type == 'lstm':
	for x in range(nb_layer - 1):
		model.add(LSTM(units=nb_neuron, activation=activation_function, dropout=0.5, input_shape=(time_step, input_dim), return_sequences=True))
	model.add(LSTM(units=nb_neuron, activation=activation_function, dropout=0.5, input_shape=(time_step, input_dim)))
elif model_type == 'bi_lstm':
	for x in range(nb_layer - 1):
		model.add(Bidirectional(LSTM(units=nb_neuron, activation=activation_function, dropout=0.5, input_shape=(time_step, input_dim), return_sequences=True)))
	model.add(Bidirectional(LSTM(units=nb_neuron, activation=activation_function, dropout=0.5, input_shape=(time_step, input_dim))))
elif model_type == 'cnn':
	for x in range(nb_layer):
		model.add(Conv1D(nb_neuron, kernel_size=embedding_dim, padding='valid', kernel_initializer='normal', activation=activation_function))
	model.add(Flatten())
elif model_type == 'cnn_lstm':
	for x in range(nb_layer - 1):
		model.add(Conv1D(nb_neuron, kernel_size=embedding_dim, padding='valid', kernel_initializer='normal', activation=activation_function))
	model.add(MaxPooling1D(pool_size=64))
	model.add(LSTM(units=nb_neuron, activation=activation_function, dropout=0.5, input_shape=(time_step, input_dim)))
model.add(Dense(units=number_of_category, activation='softmax'))
model.compile(loss=model_loss_function, optimizer=model_optimizer, metrics=['categorical_accuracy'])

# traing of the model
# param validation_split define the part of the data to be used for validation
history = model.fit(x_train_encoded, y_train_encoded, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)

# evalution of the model
score = model.evaluate(x_test_encoded, y_test_encoded, batch_size=batch_size, verbose=1)

# Save the weights
model.save_weights('model_weights.h5')

# Save the model architecture
with open('model_architecture.json', 'w') as f:
	f.write(model.to_json())

# print the result
print('train accuracy : %.3f' % float(history.history['categorical_accuracy'][epochs-1]))
print('val accuracy : %.3f' % float(history.history['val_categorical_accuracy'][epochs-1]))
print('Test accuracy: %.3f' % float(score[1]))

''' use for defining the precision of the model on emotion
cpt_tot_emotion = [0, 0, 0, 0, 0]
cpt_bonne_reponse = [0, 0, 0, 0, 0]
for x in range(len(x_test)):

	phrase = x_test[x]
	predictions = model.predict(np.array(t.texts_to_matrix([phrase], mode=texts_to_matrix_mode)))
	emotion_attendu_index = int(y_test[x])
	emotion_attendu = convert_emotion_index_to_txt(emotion_attendu_index)
	emotion_predis_index = predictions.argmax()
	emotion_predis = convert_emotion_index_to_txt(emotion_predis_index)
	cpt_tot_emotion[emotion_attendu_index] += 1
	if emotion_attendu == emotion_predis:
		cpt_bonne_reponse[emotion_attendu_index] += 1

	# print('attendu=' + emotion_attendu + ' : trouvé=' + emotion_predis + ' : ' + phrase)

print(cpt_tot_emotion)
print(cpt_bonne_reponse)
'''