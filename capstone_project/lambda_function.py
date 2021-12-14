#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

def open_image(url):
    with Image.open(url) as img:
        img = img.resize((150, 150), Image.NEAREST)
    return img

def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x

interpreter = tflite.Interpreter(model_path='xception-clothing-model.tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = ['black_dress',
 'black_pants',
 'black_shirt',
 'black_shoes',
 'black_shorts',
 'black_suit',
 'blue_dress',
 'blue_pants',
 'blue_shirt',
 'blue_shoes',
 'blue_shorts',
 'brown_hoodie',
 'brown_pants',
 'brown_shoes',
 'green_pants',
 'green_shirt',
 'green_shoes',
 'green_shorts',
 'green_suit',
 'pink_hoodie',
 'pink_pants',
 'pink_skirt',
 'red_dress',
 'red_hoodie',
 'red_pants',
 'red_shirt',
 'red_shoes',
 'silver_shoes',
 'silver_skirt',
 'white_dress',
 'white_pants',
 'white_shoes',
 'white_shorts',
 'white_suit',
 'yellow_dress',
 'yellow_shorts',
 'yellow_skirt']


def predict(url):
    img = open_image(url)

    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = preprocess_input(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    float_predictions = preds[0].tolist()

    return dict(zip(classes, float_predictions))

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result

# if __name__ == "__main__":
#     print(predict('./dataset/test_black.jpg'))

