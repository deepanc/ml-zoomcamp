#!/usr/bin/env python
# coding: utf-8


import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input
from PIL import Image
import tensorflow.lite as tflite
from keras_image_helper import create_preprocessor

model = keras.models.load_model('clothing-model.h5')

img = load_img('pants.jpg', target_size=(299, 299))
x = np.array(img)
X = np.array([x])
X = preprocess_input(X)

preds = model.predict(X)


preds

classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

dict(zip(classes, preds[0]))


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('clothing-model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)


interpreter = tflite.Interpreter(model_path='clothing-model.tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)




classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

dict(zip(classes, preds[0]))

with Image.open('pants.jpg') as img:
    img = img.resize((299, 299), Image.NEAREST)


def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x

x = np.array(img, dtype='float32')
X = np.array([x])

X = preprocess_input(X)


interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)



classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

dict(zip(classes, preds[0]))



get_ipython().system('pip install keras-image-helper')


get_ipython().system('pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime')



interpreter = tflite.Interpreter(model_path='clothing-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

preprocessor = create_preprocessor('xception', target_size=(299, 299))

url = 'http://bit.ly/mlbookcamp-pants'
X = preprocessor.from_url(url)


interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)


classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

dict(zip(classes, preds[0]))




