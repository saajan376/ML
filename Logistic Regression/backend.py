from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import logging
import traceback

# --- Initialize Logging ---
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": ["http://localhost:3000"]}})

# --- Load Model and Scaler ---
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    logging.info("Model and scaler loaded successfully.")
except FileNotFoundError as e:
    logging.error("Error: model.pkl or scaler.pkl not found.")
    model = None
    scaler = None

# --- Define Columns ---
MODEL_COLUMNS = [
    'age', 'trestbps', 'chol', 'thalch', 'oldpeak',
    'sex_Male', 'dataset_Hungarian', 'dataset_VA',
    'cp_atypical_angina', 'cp_non_anginal', 'cp_typical_angina',
    'fbs_True', 'restecg_normal', 'restecg_st-t_wave_abnormality',
    'exang_True'
]
NUMERICAL_COLS = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
REQUIRED_FIELDS = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang']

# --- Input Validation ---
def validate_input(data):
    try:
        if not (0 < float(data['age']) <= 120):
            return False, "Age must be between 1 and 120"
        if not (0 < float(data['trestbps']) <= 300):
            return False, "Trestbps must be between 1 and 300"
        # Add other validations...
        if data['sex'].lower() not in ['male', 'female']:
            return False, "Sex must be 'Male' or 'Female'"
        # Add other categorical validations...
        return True, ""
    except (ValueError, TypeError):
        return False, "Invalid data type for numerical fields"

# --- Health Check Endpoint ---
@app.route('/health', methods=['GET'])
def health():
    if not model or not scaler:
        return jsonify({'status': 'unhealthy', 'error': 'Model or scaler not loaded'}), 500
    return jsonify({'status': 'healthy'}), 200

# --- Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({'status': 'error', 'error': 'Model is not loaded.'}), 500

    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'error': 'No input data provided.'}), 400

    missing_fields = [field for field in REQUIRED_FIELDS if field not in data]
    if missing_fields:
        return jsonify({'status': 'error', 'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400

    is_valid, error_msg = validate_input(data)
    if not is_valid:
        return jsonify({'status': 'error', 'error': error_msg}), 400

    try:
        input_df = pd.DataFrame([data])
        for col in NUMERICAL_COLS:
            input_df[col] = pd.to_numeric(input_df[col])

        # One-hot encoding with normalization
        input_df['sex_Male'] = (input_df['sex'].str.lower() == 'male').astype(int)
        input_df['dataset_Hungarian'] = (input_df['dataset'].str.lower() == 'hungarian').astype(int)
        input_df['dataset_VA'] = (input_df['dataset'].str.lower() == 'va').astype(int)
        input_df['cp_atypical_angina'] = (input_df['cp'].str.lower() == 'atypical angina').astype(int)
        input_df['cp_non_anginal'] = (input_df['cp'].str.lower() == 'non-anginal').astype(int)
        input_df['cp_typical_angina'] = (input_df['cp'].str.lower() == 'typical angina').astype(int)
        input_df['fbs_True'] = (input_df['fbs'].str.lower() == 'true').astype(int)
        input_df['restecg_normal'] = (input_df['restecg'].str.lower() == 'normal').astype(int)
        input_df['restecg_st-t_wave_abnormality'] = (input_df['restecg'].str.lower() == 'st-t wave abnormality').astype(int)
        input_df['exang_True'] = (input_df['exang'].str.lower() == 'true').astype(int)

        for col in MODEL_COLUMNS:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[MODEL_COLUMNS]
        input_df[NUMERICAL_COLS] = scaler.transform(input_df[NUMERICAL_COLS])

        probability = model.predict_proba(input_df)[0][1]
        prediction_result = 1 if probability >= 0.5 else 0

        return jsonify({
            'status': 'success',
            'data': {
                'prediction': 'Has Disease' if prediction_result == 1 else 'No Disease',
                'probability': round(probability * 100, 2)
            }
        })

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        logging.error("Traceback:", exc_info=True)
        return jsonify({'status': 'error', 'error': 'An error occurred during data processing.'}), 400

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)