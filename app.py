from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# ✅ Load the trained model at the start
try:
    model = joblib.load("fraud_detection_model.pkl")
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print("🚨 Error Loading Model:", str(e))

@app.route("/")
def home():
    return "Flask App is Running! Use /predict to test the API."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("📥 Received Data:", data)  # ✅ Log received data

        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request"}), 400

        if not isinstance(data["features"], list):
            return jsonify({"error": "'features' should be a list"}), 400

        try:
            features = np.array(data["features"]).reshape(1, -1)
            print("🔄 Processed Features:", features)  # ✅ Log processed features
        except Exception as e:
            print("❌ NumPy Reshape Error:", str(e))
            return jsonify({"error": "NumPy reshape error: " + str(e)}), 400

        try:
            prediction = model.predict(features)[0]
            print("✅ Prediction:", prediction)  # ✅ Log model prediction
        except Exception as e:
            print("❌ Model Prediction Error:", str(e))
            return jsonify({"error": "Model prediction error: " + str(e)}), 400

        response = {"fraud_prediction": int(prediction)}
        print("📤 Response Sent:", response)  # ✅ Log response
        return jsonify(response)

    except Exception as e:
        print("❌ General Error:", str(e))
        return jsonify({"error": str(e)}), 400


import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use the environment-assigned port
    app.run(debug=True, host='0.0.0.0', port=port)
