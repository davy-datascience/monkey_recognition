from flask import Flask, render_template, request, redirect, url_for, session
import io
import json
import urllib
from urllib.request import urlopen, Request
import boto3
from PIL import Image
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras.utils.data_utils import get_file

from classes.prediction import Prediction, get_predictions_json
import pandas as pd


app = Flask(__name__)
app.secret_key = b"4\xbb\xa0Z_\xb1~\x991\xf7\x8c\xaa4C\xc8\xccV'\xe11\xdf\xc0\xe7\x1c"
#app.debug = True


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/predict')
def predict():
    predictions = json.loads(session['predictions'])
    return render_template('predict.html', predictions=predictions, title="Predictions")


def download_file(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = Request(url=url, headers=headers)
    fd = urllib.request.urlopen(req)
    file = io.BytesIO(fd.read())
    return file



@app.route('/', methods=['POST'])
def upload():
    labels = pd.read_csv("monkey_labels.csv")
    bucket = "monkey-recognition"
    s3 = boto3.client('s3')
    s3.download_file('monkey-recognition', 'model.h5', 'model.h5')
    model = load_model('model.h5')
    predictions = []
    for f in request.files.getlist('file'):
        # Save image to s3
        s3.put_object(Bucket="monkey-recognition", Key=f.filename, Body=f, ACL='public-read', ContentType='image/jpeg')

        img_file = download_file("https://monkey-recognition.s3.eu-west-3.amazonaws.com/" + f.filename)
        img = Image.open(img_file)
        img = img.resize((150, 150), Image.ANTIALIAS)
        x = image.img_to_array(img)
        x = x / 255.
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)

        result = np.where(classes[0] == np.amax(classes[0]))
        index_specy = result[0][0]
        prediction_label = labels["common_name"].iloc[index_specy]
        prediction = Prediction(f.filename, prediction_label)
        predictions.append(prediction)
    predictions_json = get_predictions_json(predictions)
    predictions_str = json.dumps(predictions_json)
    session['predictions'] = predictions_str
    return redirect(url_for('predict'))
