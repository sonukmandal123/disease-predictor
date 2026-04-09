from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Convert input into correct format
    input_data = []

    for column in encoders:
        value = data.get(column)

        # Convert text → number using encoder
        if value is not None:
            encoded = encoders[column].transform([value])[0]
            input_data.append(encoded)

    # Predict
    prediction = model.predict([input_data])

    # Decode disease back to name
    disease = encoders["Disease"].inverse_transform(prediction)

    return jsonify({"disease": disease[0]})

app.run(host="0.0.0.0", port=10000)