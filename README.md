# 🚀 Credit Card Fraud Detection System  

## 📌 Overview  
This project is a **machine learning-based Credit Card Fraud Detection system** that predicts whether a transaction is **fraudulent** or **valid**.  
The model is trained using **Scikit-Learn** and deployed using **Flask** with a web-based frontend.  

---

## 🔧 Tech Stack  
### **Machine Learning (Google Colab)**  
- 📊 **Scikit-Learn** (Logistic Regression, SVM, etc.)  
- ⚡ **NumPy, Pandas** (Data processing)  
- 📈 **Data Scaling** (StandardScaler)  
- 🎯 **Accuracy Calculation** (`accuracy_score`)  

### **Backend (Flask API)**  
- 🏗 **Flask** (Web framework)  
- 🌍 **Flask-CORS** (Handling frontend-backend communication)  
- 🔥 **Gunicorn** (For production deployment)  

### **Frontend (Web UI)**  
- 🎨 **HTML, CSS** (User Interface)  
- 🖥 **JavaScript (Fetch API)** (For handling user input)  

### **Deployment**  
- 🚀 **Render** (Hosting the Flask API & frontend)  
- 📂 **GitHub** (Version control & deployment)  

---

## 📊 Dataset & Features  
This model is trained on a **credit card transaction dataset** where features are **numerical values representing transaction properties**.  
Since raw data contains sensitive information, **Principal Component Analysis (PCA)** is used to anonymize the dataset.

### **Features Used in Prediction:**  
| **Feature** | **Description** |
|------------|---------------|
| `Feature 1` | **Transaction Amount (Scaled)** - Normalized value between `-2 to 2`. |
| `Feature 2` | **Spending Behavior** - Normalized spending pattern between `-1 to 1`. |
| `Feature 3` | **Location Risk** - Higher values indicate a higher fraud risk (`-3 to 3`). |

📌 **Example Input:**  
```json
{
  "features": [0.1, -0.2, 0.3]
}