import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib

app = Flask(__name__)

# Folder to save uploaded model
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Global variable to store the loaded model
categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
model = None
scaler = None
label_encoders = None

def load_model(filepath):
    """Load the machine learning model from a pickle file."""
    global model, scaler, label_encoders
    try:
        _model = joblib.load(filepath)
        model = _model['model']
        scaler = _model['scaler']
        label_encoders = _model['label_encoders']
        return "Model loaded successfully."
    except Exception as e:
        return f"Error loading model: {e}"


# Load default model if exists
default_model_path = "loan_default_full_model.pkl"
if os.path.exists(default_model_path):
    print(load_model(default_model_path))


@app.route('/')
def home():
    return render_template("index.html", prediction=None, message=None)


@app.route('/upload', methods=['POST'])
def upload_model():
    """Endpoint for uploading a .pkl model file."""
    if 'file' not in request.files:
        return render_template("index.html", prediction=None, message="No file part")

    file = request.files['file']

    if file.filename == '':
        return render_template("index.html", prediction=None, message="No selected file")

    if not file.filename.endswith('.pkl'):
        return render_template("index.html", prediction=None, message="Invalid file format. Please upload a .pkl file.")

    # Save file to uploads directory
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], "model.pkl")
    file.save(filepath)

    # Load the model
    load_message = load_model(filepath)
    return render_template("index.html", prediction=None, message=load_message)


def get_index_from_value(key: str, value: str) -> float:
    options = {
        "education": ["highschool", "bachelor", "masters", "phd"],
        "employmentType": ["fulltime", "parttime", "freelance", "contract"],
        "maritalStatus": ["single", "married", "divorced", "widowed"],
        "hasMortgage": ["yes", "no"],
        "hasDependents": ["yes", "no"],
        "loanPurpose": ["education", "home", "car", "medical", "business"],
        "hasCosigner": ["yes", "no"]
    }

    if key in options and value in options[key]:
        return float(options[key].index(value))

    return None  # Return -1.0 if the key or value is not found


def preprocess_input(data):
    input_features = []

    for key, value in data.items():
        if key in categorical_cols:  # Check if the feature is categorical
            if value in label_encoders[key].classes_:
                encoded_value = label_encoders[key].transform([value])[0]
            else:
                print(
                    f"Warning: '{value}' is not in training data for '{key}'. Assigning default category.")
                default_category = label_encoders[key].classes_[
                    0]  # Assign most frequent category
                encoded_value = label_encoders[key].transform([default_category])[
                    0]
            input_features.append(encoded_value)
        else:
            try:
                input_features.append(float(value))
            except ValueError:
                print(
                    f"Warning: Could not convert '{value}' to float for feature '{key}'. Using 0.")
                input_features.append(0)

    features = np.array(input_features).reshape(1, -1)

    return features


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions with the uploaded model."""
    global model
    data = request.form.to_dict()
    try:
        features = preprocess_input(data)

        # Scale numerical features
        features = scaler.transform(features)

        # Predict
        # prediction = model]'model'].predict(features)[0]
        probability = model.predict_proba(features)[:, 1][0]

        return render_template("index.html", prediction=str(probability), message=None)

    except Exception as e:
        return render_template("index.html", prediction=None, message=f"Prediction failed: {e}")


def predt():
    # Extracting features from form data correctly
    input_features = [0.5]*16
    features = np.array(input_features).reshape(1, -1)
    prediction = model.predict(features)[0]
    prediction = str(prediction)


if __name__ == '__main__':
    app.run(debug=True)
