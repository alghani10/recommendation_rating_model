import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
import requests
from flask import Flask, request, jsonify

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load recommendation_rating_model.h5
model = tf.keras.models.load_model("recommendation_rating_model.h5")
# Label yang ada di dalam recommendation_rating_model.h5
label = ["Epochs","Training Error"]

app = Flask(__name__)

@app.route("recommendation_rating_model.h5")
def hello():
    """Returns a greeting"""
    return "Hello, World!"

@app.route("/recommendation_rating_model.h5/predict/<int:id>", methods=["GET","POST"])
def predict(id):
    """Makes a prediction for a user and returns recommended places"""
    id_place = range(1, 20)
    # Creating dataset for making recommendations for the first user
    tourism_data = np.array(list(set(id_place)))
    tourism_data[:10]

    user_id = id

    user = np.array([user_id for i in range(len(tourism_data))])
    user[:10]

    predictions = model.predict([user, tourism_data])

    predictions = np.array([a[0] for a in predictions])

    recommended_tourism_ids = (predictions).argsort()[:10]
    list_recommended_tourism_ids = recommended_tourism_ids.tolist()

    # get data from api to convert id to place id
    place_id = []
    for i in list_recommended_tourism_ids:
        r = requests.get(
            f'https://retrip-connect-your-trip-default-rtdb.asia-southeast1.firebasedatabase.app/destinasi.json?auth=lOQHLNGXFVsZ4Pydd7Bn9HUjCDrrHNqflzsYLIA2')
        s = r.json()
        place_id.append(s['data'])

    response_json = {
        "data": user_id,
        "prediction": place_id,
        "category": " ",
        "city": " ",
        "description": " ",
        "img": " ",
        "lang": " ",
        "lat": " ",
        "name": " ",
    }

    # return a response in json format
    return jsonify(response_json), 200
