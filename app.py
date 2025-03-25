from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from flask_cors import CORS

# Load the trained model
model = joblib.load('fraud_detection_model.pkl')

# Initialize Flask app
app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return render_template("index.html")
    # return "Credit Card Fraud Detection API is running!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received Data:", data)  # ✅ Log received data

        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request"}), 400

        if not isinstance(data["features"], list):
            return jsonify({"error": "'features' should be a list"}), 400

        try:
            features = np.array(data["features"]).reshape(1, -1)
            print("Processed Features:", features)  # ✅ Log processed features
        except Exception as e:
            print("NumPy Reshape Error:", str(e))  # ✅ Log NumPy error
            return jsonify({"error": "NumPy reshape error: " + str(e)}), 400

        try:
            prediction = model.predict(features)[0]
            print("Prediction:", prediction)  # ✅ Log prediction
        except Exception as e:
            print("Model Prediction Error:", str(e))  # ✅ Log model error
            return jsonify({"error": "Model prediction error: " + str(e)}), 400

        response = {"fraud_prediction": int(prediction)}
        print("Response Sent:", response)  # ✅ Log response

        return jsonify(response)

    except Exception as e:
        print("General Error:", str(e))  # ✅ Log general error
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)