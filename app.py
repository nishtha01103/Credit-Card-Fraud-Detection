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
        # Get JSON data from request
        data = request.get_json()
        print("Received Data:", data)

        # Convert to NumPy array and reshape
        features = np.array(data['features']).reshape(1, -1)

        # Predict
        prediction = model.predict(features)[0]

        response = {"fraud_prediction": int(prediction)}
        print("Response Sent:", response)

        # Return result
        return jsonify(response)
        # return render_template("index.html",prediction_text="The Prediction is ")

    except Exception as e:
        return jsonify({'error': str(e)}),400


if __name__ == '__main__':
    app.run(debug=True)