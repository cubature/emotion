import tensorflow as tf
import pickle
import numpy as np
import tools as tools
from tensorflow.keras.models import Sequential, model_from_json, load_model
from keras.preprocessing.sequence import pad_sequences


# -----  Loading the trained model   -------
# step 1 : load the stored structure
with open('model_architecture.json', 'r') as f:
    model = model_from_json(f.read())
# step 2 : Load weights into the new model
model.load_weights('model_weights.h5')

# load the tokenizer to encode the sentence
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# enter the sentence
phrase = "So sleepy again and it's not even that late. I fail once again."#input("Enter the sentence to be tested :\n")

# prediction with the model

# 1 bag_of_words and not rnn, lstm or bi-lstm
# encodedInput = tokenizer.texts_to_matrix([phrase], mode='count')[:, 1:]

# 2 bag_of_words and cnn, rnn, lstm or bi-lstm
# encodedInput = tokenizer.texts_to_matrix([phrase], mode='count')[:, 1:]
# encodedInput = np.reshape(encodedInput, (encodedInput.shape[0], 1, encodedInput.shape[1]))

# 3 bag_of_words and cnn_lstm
# encodedInput = tokenizer.texts_to_matrix([phrase], mode='count')[:, 1:]
# encodedInput = np.reshape(encodedInput, (encodedInput.shape[0], encodedInput.shape[1], 1))

# 4 sequence_of_words
# encodedInput = tokenizer.texts_to_sequences(phrase)
# encodedInput = pad_sequences(encodedInput, maxlen=40, padding='post', truncating='post')
# encodedInput = np.reshape(encodedInput, (encodedInput.shape[0], encodedInput.shape[1], 1))

# 5 glove and skip-grams
encodedInput = pad_sequences(tokenizer.texts_to_sequences(phrase), maxlen=1000)

predictions = model.predict(encodedInput)
print('predictions : ' + str(predictions))
predict_classes = model.predict_classes(encodedInput)
print('predict classes: ' + str(predict_classes))
predict_class = np.argmax(np.bincount(predict_classes))
print('predict class: ' + str(predict_class))

# analyse of the prediction
# emotion_predicted_index = predictions.argmax()
emotion_predicted = tools.convert_emotion_index_to_txt(predict_class)
print('result of the analyse: ' + emotion_predicted)
