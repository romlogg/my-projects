
import os
import argparse
import numpy as np
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # удаление сообщений от tensorflow
import tensorflow as tf
from tensorflow import keras


# Создание парсера и аргументов
parser = argparse.ArgumentParser(description="captchas recognition")
parser.add_argument("-i", dest="image", required=True, help="path to input image file")
parser.add_argument("-m", dest="model", required=True, help="path to trained model")
parser.add_argument("-j", dest="json", required=True, help="path to json")
args = parser.parse_args()
img_path = args.image
model_path = args.model
json_path = args.json


def Captcha_to_text(img_path,model_path,json_path):

    # загрузка модели и json
    m = keras.models.load_model(model_path)
    with open(json_path, 'r') as json_file:
        num_to_char = json.load(json_file)

    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.transpose(img, perm=[1, 0, 2])
    img = img.numpy()
    img = np.expand_dims(img, axis = 0)

    test_pred = m.predict(img)

    score = []
    for i in range(test_pred.shape[1]):
        arr = test_pred[0][i]
        max_index = np.argmax(arr)
        if max_index == test_pred.shape[2]-1:
            continue
        max_value = np.max(arr)
        score.append(max_value)

    if min(score) < 0.60:
        return "error_recognize"

    test_pred = keras.backend.ctc_decode(test_pred, input_length=np.ones(img.shape[0])*int(img.shape[1]/4), greedy=True)
    test_pred = test_pred[0][0][0:img.shape[0],0:7].numpy()

    answers = ["".join(list(map(lambda x:num_to_char[str(x)], label))).replace("UKN",'') for label in test_pred]
    
    return answers, min(score)

print(Captcha_to_text(img_path,model_path,json_path))
