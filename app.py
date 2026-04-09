from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # ✅ allow frontend connection

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# ✅ IMPORTANT: correct feature order (MUST MATCH TRAINING)
FEATURE_ORDER = [
    "Fever",
    "Cough",
    "Fatigue",
    "Difficulty Breathing",
    "Age",
    "Gender",
    "Blood Pressure",
    "Cholesterol Level"
]

# ✅ Home route (no more "Not Found")
@app.route("/")
def home():
    return "✅ Disease Prediction API is running"

# ✅ Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        print("Received data:", data)

        input_data = []

        # ✅ FIX: enforce correct order
        for col in FEATURE_ORDER:
            if col not in data:
                return jsonify({"error": f"Missing field: {col}"}), 400

            value = data[col]

            # Age is numeric, not label-encoded
            if col == "Age":
                input_data.append(int(value))
            else:
                # Encode using saved encoder
                encoded = encoders[col].transform([value])[0]
                input_data.append(encoded)

        print("Encoded input:", input_data)

        # Predict
        prediction = model.predict([input_data])

        # Decode result
        disease = encoders["Disease"].inverse_transform(prediction)

        return jsonify({
            "disease": disease[0],
            "status": "success"
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500


# ✅ FIX for Render deployment
port = int(os.environ.get("PORT", 10000))
app.run(host="0.0.0.0", port=port)