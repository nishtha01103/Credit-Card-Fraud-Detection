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

        # Check if 'features' key is present in the received JSON
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request"}), 400

        # Ensure 'features' is a list
        if not isinstance(data["features"], list):
            return jsonify({"error": "'features' should be a list"}), 400

        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)[0]

        response = {"fraud_prediction": int(prediction)}
        print("Response Sent:", response)  # ✅ Log response

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}),400


if __name__ == '__main__':
    app.run(debug=True)