
import os
import numpy as np
import pandas as pd
import json
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # удаление сообщений от tensorflow
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.layers import Activation, MaxPooling2D, Flatten, Conv2D, Dropout, Dense
from tensorflow.keras import layers


# Создание парсера и аргументов
parser = argparse.ArgumentParser(description="captchas model training")
parser.add_argument("-f", dest="folder", required=True, help="path to input folder")
parser.add_argument("-m", dest="model", required=True, help="path to trained model")
parser.add_argument("-j", dest="json", required=True, help="path to json")
args = parser.parse_args()
folder_path = args.folder
model_path = args.model
json_path = args.json

filenames = os.listdir(folder_path)


# создание DF и Извлечение символов
text = []
for i in range(len(filenames)):
    text.append(filenames[i][:len(filenames[i])-4])
df = pd.DataFrame({"filename": filenames,"text": text})

input = {row.text: row.filename for row in df.itertuples()}
characters = sorted(set(''.join(input.keys())))
char_to_num = {v: i for i, v in enumerate(characters)}
num_to_char = {str(i): v for i, v in enumerate(characters)}
num_to_char['-1'] = 'UKN'


# Функция перевода изображений обучающей выборки в массив
def encode_single_sample(filename):
    img_path = os.path.join(folder_path, filename)
    img = tf.io.read_file(img_path)

    try:
      img = tf.io.decode_png(img, channels=3)
    except Exception as e:
      print(img_path)
      raise e

    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.transpose(img, perm=[1, 0, 2])

    return img.numpy()


# Разбивка на обучение и валидацию, перевод в нампай массив
X, y = [],[]
marriage = []
train_dataset = list(input.items())
patt = encode_single_sample(train_dataset[0][1]).shape
for i in train_dataset:
    nump = encode_single_sample(i[1])
    if nump.shape == patt:
        X.append(encode_single_sample(i[1]))
        y.append(list(map(lambda x:char_to_num[x], i[0])))
    else:
        marriage.append(i[1])


X = np.asarray(X)
y = tf.keras.preprocessing.sequence.pad_sequences(np.asarray(y, dtype=object), 7, padding='post', value=-1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)


# Построение модели
class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        labels_mask = 1 - tf.cast(tf.equal(y_true, -1), dtype="int64")
        labels_length = tf.reduce_sum(labels_mask, axis=1)
        loss = self.loss_fn(y_true, y_pred, input_length, tf.expand_dims(labels_length, -1))
        self.add_loss(loss)

        return y_pred

def build_model():
    x_shape = X[0].shape
    x_target_shape = (int(x_shape[0]/4), int(x_shape[1]/4)*64)
    input_img = layers.Input(shape=x_shape, name="image", dtype="float32")
    labels = layers.Input(name="label", shape=(7, ), dtype="float32")

    x = layers.Conv2D(32,(3, 3),activation="relu",kernel_initializer="he_normal",padding="same",name="Conv1")(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Conv2D(64,(3, 3),activation="relu",kernel_initializer="he_normal",padding="same",name="Conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Reshape(target_shape = x_target_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
    x = layers.Dense(len(characters)+1, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)
    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="ocr_cnn_lstm_model")
    model.compile(optimizer=keras.optimizers.Adam())

    return model

model = build_model()

# Преобразование данных на вход к модели
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.map(lambda x,y: {'image':x, 'label':y}).batch(16).prefetch(buffer_size=tf.data.AUTOTUNE)

validation_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
validation_dataset = validation_dataset.map(lambda x,y: {'image':x, 'label':y}).batch(16).prefetch(buffer_size=tf.data.AUTOTUNE)


# Обучение
EPOCH = 70 

early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCH, callbacks=[early_stopping],)

# Cохранение модели и json
prediction_model = keras.models.Model(model.get_layer(name="image").input, model.get_layer(name="dense2").output)
prediction_model.save(model_path+"/"+"model.h5")

json_data = json.dumps(num_to_char)
with open(json_path+"/"+"num_to_char.json", 'w') as json_file:
    json_file.write(json_data)


# Проверка качества на Валидации
y_pred = prediction_model.predict(X_val)
y_pred = keras.backend.ctc_decode(y_pred, input_length=np.ones(X_val.shape[0])*int(X_val.shape[1]/4), greedy=True)
y_pred = y_pred[0][0][0:X_val.shape[0],0:7].numpy()

compute_perf_metric = np.sum(y_pred == y_val)/(y_pred.shape[0]*y_pred.shape[1])


if marriage:
    print("Не все изображения одинакового размера, проверьте изображения ", marriage)

print("Качество обученной модели", compute_perf_metric)

if compute_perf_metric < 0.7:
    print("Качество низкое, увеличите количество размеченных изображений")