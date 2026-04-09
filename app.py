from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# ✅ Home route
@app.route("/")
def home():
    return "✅ Disease Prediction API is running"

# ✅ Debug route — shows encoder keys and model expected features
@app.route("/debug")
def debug():
    encoder_keys = list(encoders.keys())
    try:
        n_features = model.n_features_in_
    except:
        n_features = "unknown"
    return jsonify({
        "encoder_keys": encoder_keys,
        "model_expected_features": n_features
    })

# ✅ Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        print("Received data:", data)

        # Get the feature columns from encoders (excluding Disease which is the target)
        feature_cols = [k for k in encoders if k != "Disease"]
        
        input_data = []

        for col in feature_cols:
            value = data.get(col)
            if value is None:
                return jsonify({"error": f"Missing field: {col}", "expected_fields": feature_cols}), 400

            encoded = encoders[col].transform([value])[0]
            input_data.append(encoded)

        print("Encoded input:", input_data, "length:", len(input_data))

        # If model expects more features than we have encoders for,
        # check if Age is a numeric column that wasn't label-encoded
        n_expected = getattr(model, 'n_features_in_', len(input_data))
        if n_expected > len(input_data):
            # Try inserting Age as numeric at position after "Difficulty Breathing" (index 4)
            age_val = int(data.get("Age", 25))
            insert_pos = min(4, len(input_data))
            input_data.insert(insert_pos, age_val)
            print("Added Age at position", insert_pos, "=> new length:", len(input_data))

        # Predict
        prediction = model.predict([input_data])

        # Decode result
        disease = encoders["Disease"].inverse_transform(prediction)

        return jsonify({
            "disease": disease[0],
            "status": "success"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500


# ✅ Render deployment
port = int(os.environ.get("PORT", 10000))
app.run(host="0.0.0.0", port=port)