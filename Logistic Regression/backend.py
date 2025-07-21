# --- Step 1: Import Necessary Libraries ---
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import traceback # Import traceback to get detailed error information

# --- Step 2: Initialize the Flask Application ---
app = Flask(__name__)
CORS(app)

# --- Step 3: Load the Pre-trained Model and Scaler ---
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: model.pkl or scaler.pkl not found. Please run the training script to generate these files.")
    model = None
    scaler = None

# --- Step 4: Define the Feature Columns ---
# **FIX:** Corrected column names to use underscores instead of spaces, matching pandas get_dummies() output.
MODEL_COLUMNS = [
    'age', 'trestbps', 'chol', 'thalch', 'oldpeak',
    'sex_Male', 'dataset_Hungarian', 'dataset_VA',
    'cp_atypical_angina', 'cp_non_anginal', 'cp_typical_angina',
    'fbs_True', 'restecg_normal', 'restecg_st-t_wave_abnormality',
    'exang_True'
]
NUMERICAL_COLS = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']


# --- Step 5: Create the Prediction API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({'error': 'Model is not loaded. Please check server logs.'}), 500

    data = request.get_json()

    try:
        # --- Data Preprocessing ---
        input_df = pd.DataFrame([data])

        # Explicitly convert numerical columns to a numeric type.
        for col in NUMERICAL_COLS:
            input_df[col] = pd.to_numeric(input_df[col])

        # **FIX:** Use the corrected column names with underscores in the one-hot encoding logic.
        input_df['sex_Male'] = (input_df['sex'] == 'Male').astype(int)
        input_df['dataset_Hungarian'] = (input_df['dataset'] == 'Hungarian').astype(int)
        input_df['dataset_VA'] = (input_df['dataset'] == 'VA').astype(int)
        input_df['cp_atypical_angina'] = (input_df['cp'] == 'atypical angina').astype(int)
        input_df['cp_non_anginal'] = (input_df['cp'] == 'non-anginal').astype(int)
        input_df['cp_typical_angina'] = (input_df['cp'] == 'typical angina').astype(int)
        input_df['fbs_True'] = (input_df['fbs'] == 'True').astype(int)
        input_df['restecg_normal'] = (input_df['restecg'] == 'normal').astype(int)
        input_df['restecg_st-t_wave_abnormality'] = (input_df['restecg'] == 'st-t wave abnormality').astype(int)
        input_df['exang_True'] = (input_df['exang'] == 'True').astype(int)

        # Ensure all required model columns are present
        for col in MODEL_COLUMNS:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match the model's training order
        input_df = input_df[MODEL_COLUMNS]

        # --- Feature Scaling ---
        input_df[NUMERICAL_COLS] = scaler.transform(input_df[NUMERICAL_COLS])

        # --- Make Prediction ---
        probability = model.predict_proba(input_df)[0][1]
        prediction_result = 1 if probability >= 0.5 else 0

        # --- Send Response ---
        return jsonify({
            'prediction': 'Has Disease' if prediction_result == 1 else 'No Disease',
            'probability': round(probability * 100, 2)
        })

    except Exception as e:
        # Print a detailed traceback to the console for easier debugging.
        print(f"An error occurred during prediction: {e}")
        print("Traceback:")
        traceback.print_exc()
        return jsonify({'error': 'An error occurred during data processing.'}), 400

# --- Step 6: Run the Flask Application ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
