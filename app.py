from flask import Flask, render_template, request, redirect, url_for, session
import json
import configparser

from monkey_recognition.monkey_recognition import get_monkey_prediction

config = configparser.ConfigParser()
config.read("config.ini")

app = Flask(__name__)
app.secret_key = config['flask']['secret_key']
app.config['UPLOAD_FOLDER'] = config['flask']['upload_folder']
app.debug = True


@app.route('/')
def index():
    return redirect(url_for('monkeys'))


@app.route('/monkeys')
def monkeys():
    return render_template('monkey_recognition/home.html')


@app.route('/monkeys-predict')
def monkeys_predict():
    predictions = json.loads(session['predictions'])
    return render_template('monkey_recognition/predict.html', predictions=predictions, title="Predictions")


@app.route('/monkeys', methods=['POST'])
def monkeys_upload():
    session['predictions'] = get_monkey_prediction()
    return redirect(url_for('monkeys_predict'))
