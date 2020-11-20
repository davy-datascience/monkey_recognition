from json import JSONEncoder
from bson.json_util import loads


class Prediction:
    def __init__(self, img, prediction, accuracy):
        self.img = img
        self.prediction = prediction
        self.accuracy = accuracy


class PredictionEncoder(JSONEncoder):
    def default(self, obj):
        return obj.__dict__


def get_predictions_json(predictions):
    return loads(PredictionEncoder().encode(predictions))
