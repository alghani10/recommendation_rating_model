import os
import tensorflow
import numpy as np
from keras.models import load_model
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load recommendation_rating_model.h5
model = load_model("recommendation_rating_model.h5")
# Label yang ada di dalam recommendation_rating_model.h5
label = ["Epochs", "Training Error"]

@app.route("/predict", methods=["GET", "POST"])
def index():
    file = request.files.get('file')
    if file is None or file.filename == "":
        return jsonify({"error":"no file"})
    
    # Call the predict function with the user id from the file
    user_id = 1 # Replace this with the user id from the file
    response_json = predict(user_id)
    
    return jsonify(response_json)

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

    recommended_tourism_ids = predictions.argsort()[:10]
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
        "category": "",
        "city": "",
        "description": "",
        "img": "",
        "lang": "",
        "lat": "",
        "name": ""
    }
    
    return response_json

if __name__ == "__main__":
    app.run()
