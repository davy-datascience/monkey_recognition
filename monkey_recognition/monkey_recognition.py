from flask import request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from monkey_recognition.classes.prediction import Prediction, get_predictions_json
import json
import numpy as np
import pandas as pd
import os
import configparser

config = configparser.ConfigParser()
config.read("config.ini")
upload_folder = config['flask']['upload_folder']


def get_monkey_prediction():
    labels = pd.read_csv("monkey_recognition/monkey_labels.csv")
    model = load_model('monkey_recognition/model.h5')
    predictions = []
    for uploaded_file in request.files.getlist('file'):
        if uploaded_file.filename != '':
            file_path = os.path.join(upload_folder, uploaded_file.filename)
            uploaded_file.save(file_path)
            img = image.load_img(file_path, target_size=(150, 150))
            x = image.img_to_array(img)
            x = x / 255.
            x = np.expand_dims(x, axis=0)

            images = np.vstack([x])
            classes = model.predict(images, batch_size=10)

            result = np.where(classes[0] == np.amax(classes[0]))
            index_specy = result[0][0]
            prediction_label = labels["common_name"].iloc[index_specy]
            prediction = Prediction(uploaded_file.filename, prediction_label)
            predictions.append(prediction)
    predictions_json = get_predictions_json(predictions)
    predictions_str = json.dumps(predictions_json)
    return predictions_str

