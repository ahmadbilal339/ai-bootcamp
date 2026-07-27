from flask import Flask, request, jsonify
import joblib
import json

app = Flask(__name__)

# Load the latest model
model = joblib.load("model_v3.joblib")

# Load metadata
with open("model_metadata.json", "r") as file:
    metadata = json.load(file)


# Endpoint 1: Return model metadata
@app.route("/models", methods=["GET"])
def get_models():
    return jsonify(metadata)


# Endpoint 2: Make predictions
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = data["features"]

    predictions = model.predict(features)

    return jsonify({
        "predictions": predictions.tolist()
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)