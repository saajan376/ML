import flask
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
from flask_cors import CORS

app= Flask(__name__)
CORS(app)

model=None
model_columns=None
label_encoder=None

def train_model():
    global model, model_columns, label_encoder
    try:
        df=pd.read_csv('mushrooms.csv')
        print("Dataset loaded successfully for training.")
    except FileNotFoundError:
        print("Error: 'mushrooms.csv' not found. Cannot train model.")
        return

    x=df.drop('class', axis=1)
    y=df['class']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    x_encoded = pd.get_dummies(x)
    model_columns = x_encoded.columns

    print("Training the Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(x_encoded, y_encoded)
    print("Model training complete!")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not trained yet.'}), 503
    try:
        data = request.get_json()
        query_df= pd.DataFrame([data])
        query_encoded = pd.get_dummies(query_df).reindex(columns=model_columns, fill_value=0)

        prediction_encoded = model.predict(query_encoded)
        prediction_proba = model.predict_proba(query_encoded)
        prediction_text = label_encoder.inverse_transform(prediction_encoded)[0]

        if prediction_text == 'p':
            result_class = 'Poisonous'
            confidence = prediction_proba[0][1] # Probability of being poisonous
        else:
            result_class = 'Edible'
            confidence = prediction_proba[0][0]