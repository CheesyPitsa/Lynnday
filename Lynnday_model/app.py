from __future__ import division, print_function
import glob
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
import numpy as np
import pandas as pd
import os
from glob import glob
import tensorflow as tf

# для flask
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# загрузка весов и модели
base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(80, 80, 3))

# структура модели
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(2048,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# 3 класса - nude, safe и sexy
model.add(Dense(3, activation='softmax'))
# загрузка весов
model.load_weights("models/weight_80_36000.hdf5")
# компиляция модели
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


def predict_if_safe(path):
    images = glob("uploads/*")
    try:
        prediction_images = []
        for i in range(len(images)):
            image = tf.keras.utils.load_img(images[i], target_size=(80, 80, 3))
            print(images[i])
            image = tf.keras.utils.img_to_array(image)
            image = image/255
            prediction_images.append(image)
        prediction_images = np.array(prediction_images)
        prediction_images = base_model.predict(prediction_images)
        prediction_images = prediction_images.reshape(prediction_images.shape[0], 2 * 2 * 512)
        prediction_array = np.argmax(model.predict(prediction_images), axis=-1)

        train = pd.read_csv('train_new.csv')
        y = train['class']
        y = pd.get_dummies(y)
        result = y.columns.values[prediction_array][0]
    except:
        result = "Пожалуйста, выберите корректный файл"

    files = glob('uploads/*')
    for file in files:
        os.remove(file)
    if str(result) == "nude":
        result = "Изображение неприемлимо для трансляции."
    elif str(result) == 'sexy':
        result = "Изображение не рекомендуется для трансляции."
    elif str(result) == 'safe':
        result = "Изображение безопасно и подходит для трансляции."

    return str(result)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(file.filename))
        file.save(file_path)
        result = predict_if_safe(file_path)
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
