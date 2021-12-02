#!/usr/bin/env python
# coding: utf-8


import numpy as np
import tensorflow as tf
from tensorflow import keras
# import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
from PIL import Image

tf.__version__
model = keras.models.load_model('dogs_cats_10_0.687.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('cats-dogs-v2.tflite', 'wb') as f_out:
    f_out.write(tflite_model)

interpreter = tflite.Interpreter(model_path='cats-dogs-v2.tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_input(x):
    x /= 255
    x -= 1.
    return x

def predict(url):
    img = download_image(url)
    prepared_img = prepare_image(img, (150, 150))
    x = np.array(prepared_img, dtype='float32')
    X = np.array([x])
    X = preprocess_input(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    return preds

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result


