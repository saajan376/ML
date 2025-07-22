import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# --- 0. Initialize Flask App ---
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) to allow frontend requests
CORS(app)


# --- Global Variables for Model and Preprocessing Info ---
# We will train the model once when the server starts and store it in memory.
model = None
# We also need to store the column order from the one-hot encoding
model_columns = None
label_encoder = None

def train_model():
    """
    This function loads the data, preprocesses it, and trains the
    logistic regression model. It's called once when the server starts.
    """
    global model, model_columns, label_encoder

    # --- 1. Load and Preprocess Data ---
    try:
        df = pd.read_csv('mushrooms.csv')
        print("Dataset loaded successfully for training.")
    except FileNotFoundError:
        print("Error: 'mushrooms.csv' not found. Cannot train model.")
        return

    # Separate Features (X) and Target (y)
    X = df.drop('class', axis=1)
    y = df['class']

    # Encode the Target Variable (y)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Encode the Feature Variables (X) using One-Hot Encoding
    X_encoded = pd.get_dummies(X)
    # Store the column names and order from the training data
    model_columns = X_encoded.columns

    # --- 2. Build and Train the Model ---
    print("Training the Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_encoded, y_encoded)
    print("Model training complete!")


# --- 3. Define the Prediction API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles prediction requests from the frontend.
    """
    if model is None:
        return jsonify({'error': 'Model is not trained yet. Please wait.'}), 503

    try:
        # Get the JSON data sent from the frontend
        json_data = request.get_json()
        # Convert the incoming JSON to a pandas DataFrame
        query_df = pd.DataFrame([json_data])

        # IMPORTANT: Reindex the query DataFrame to match the model's training columns.
        # This ensures that all one-hot encoded columns are present and in the correct order.
        # `fill_value=0` sets any missing columns (features not present in the query) to 0.
        query_encoded = pd.get_dummies(query_df).reindex(columns=model_columns, fill_value=0)

        # Make a prediction and get the probability
        prediction_encoded = model.predict(query_encoded)
        prediction_proba = model.predict_proba(query_encoded)

        # Decode the prediction back to the original label ('e' or 'p')
        prediction_text = label_encoder.inverse_transform(prediction_encoded)[0]
        
        # Determine the final class and confidence
        if prediction_text == 'p':
            result_class = 'Poisonous'
            confidence = prediction_proba[0][1] # Probability of being poisonous
        else:
            result_class = 'Edible'
            confidence = prediction_proba[0][0] # Probability of being edible

        # Return the result as a JSON response
        return jsonify({
            'prediction': result_class,
            'confidence': f"{confidence:.2f}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# --- 4. Run the Flask App ---
if __name__ == '__main__':
    # Train the model on startup
    train_model()
    # Run the Flask server
    # It will be accessible on http://127.0.0.1:5000
    app.run(port=5000, debug=True)

