#import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import datasets, layers, models

import warnings
warnings.filterwarnings("ignore")

#Load dataset
DATASET = "/content/alice_in_wonderland.txt"

text = open(DATASET, 'rb').read().decode(encoding='utf-8')
print(f'Length of text: {len(text)} Characters')

text[:250]

vocab = sorted(set(text))
print(f'Unique Characters are {len(vocab)}')


char_to_index = {u: i for i, u in enumerate(vocab)}
char_to_index

index_to_char = np.array(vocab)
index_to_char

text_as_int = np.array([char_to_index[c] for c in text])
text_as_int

print(f'{text[:30]} ------ charcaters mapped to integar ----> {text_as_int[:30]}')



SEQ_LENGTH = 40
STEP_SIZE = 3

sentences = []
next_char = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i : i + SEQ_LENGTH])
    next_char.append(text[i + SEQ_LENGTH])
x = np.zeros((len(sentences), SEQ_LENGTH, len(vocab)))
y = np.zeros((len(sentences), len(vocab)))

for i, satz in enumerate(sentences):
    for t, char in enumerate(satz):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_char[i]]] = 1


#Model Training
#Creating Batches
model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(vocab))))
model.add(Dense(len(vocab)))
model.add(Activation('softmax'))
model.compile('adam', loss="categorical_crossentropy")
model.summary()


model.fit(x, y, batch_size=256, epochs=11)


#Making Predictions
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
import random

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ""
    sentence = text[start_index : start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x_predictions = np.zeros((1, SEQ_LENGTH, len(vocab)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated
print(generate_text(300, 0.2))

#Saving the Model
model.save('model/textgeneration.h5')

