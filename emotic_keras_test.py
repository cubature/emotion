import tensorflow as tf
import pickle
import numpy as np
import tools as tools
from tensorflow.keras.models import Sequential, model_from_json, load_model


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
phrase = input("Saisissez la phrase à tester :\n")

# prediction with the model
predictions = model.predict(np.array(tokenizer.texts_to_matrix([phrase], mode='count')))
print('predictions : ' + str(predictions))

# analyse of the prediction
emotion_predicted_index = predictions.argmax()
emotion_predicted = tools.convert_emotion_index_to_txt(emotion_predicted_index)
print('résultat de l\'analyse : ' + emotion_predicted)
