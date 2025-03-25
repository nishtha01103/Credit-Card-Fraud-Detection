from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ✅ Load the trained model
try:
    model = joblib.load("fraud_detection_model.pkl")
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print("🚨 Error Loading Model:", str(e))

# ✅ Serve Frontend (index.html)
@app.route("/")
def home():
    return render_template("index.html")  # This will serve the frontend

# ✅ API Endpoint for Fraud Detection
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("📥 Received Data:", data)

        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request"}), 400

        if not isinstance(data["features"], list):
            return jsonify({"error": "'features' should be a list"}), 400

        features = np.array(data["features"]).reshape(1, -1)
        print("🔄 Processed Features:", features)

        prediction = model.predict(features)[0]
        print("✅ Prediction:", prediction)

        response = {"fraud_prediction": int(prediction)}
        print("📤 Response Sent:", response)
        return jsonify(response)

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)}), 400

# ✅ Ensure Flask Uses Render’s Dynamic Port
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)