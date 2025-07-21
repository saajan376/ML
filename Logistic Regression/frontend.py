import streamlit as st
import pandas as pd
import requests # Library to make HTTP requests
import json
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Styling ---
# You can customize the colors and styles here
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .sidebar .sidebar-content {
        background: #161A25;
    }
    .stButton>button {
        color: white;
        background-color: #007BFF;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar for User Inputs ---
st.sidebar.header("Patient Information")
st.sidebar.markdown("Enter the patient's details below to get a prediction.")

def user_input_features():
    """Creates sidebar widgets and returns a dictionary of user inputs."""
    age = st.sidebar.slider('Age', 20, 90, 54)
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    cp = st.sidebar.selectbox('Chest Pain Type', ('asymptomatic', 'atypical angina', 'non-anginal', 'typical angina'))
    trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 80, 200, 130)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 100, 600, 250)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ('False', 'True'))
    restecg = st.sidebar.selectbox('Resting ECG Results', ('lv hypertrophy', 'normal', 'st-t wave abnormality'))
    thalch = st.sidebar.slider('Max Heart Rate Achieved', 60, 220, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', ('False', 'True'))
    oldpeak = st.sidebar.slider('ST Depression (oldpeak)', 0.0, 6.2, 1.0, 0.1)
    dataset = st.sidebar.selectbox('Origin Dataset', ('Cleveland', 'Hungarian', 'VA'))

    # Store all inputs in a dictionary
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalch': thalch,
        'exang': exang,
        'oldpeak': oldpeak,
        'dataset': dataset
    }
    return data

input_data = user_input_features()

# --- Main Panel Display ---
st.title("❤️ Heart Disease Prediction")
st.markdown("This application connects to a backend server to predict the likelihood of heart disease.")
st.markdown("---")

# Display user inputs in an organized layout
st.subheader("Patient Data Entered:")
col1, col2, col3 = st.columns(3)
# Displaying the data in columns for a cleaner look
with col1:
    st.info(f"**Age:** {input_data['age']}")
    st.info(f"**Sex:** {input_data['sex']}")
    st.info(f"**Chest Pain:** {input_data['cp']}")
    st.info(f"**Dataset:** {input_data['dataset']}")
with col2:
    st.info(f"**Resting BP:** {input_data['trestbps']}")
    st.info(f"**Cholesterol:** {input_data['chol']}")
    st.info(f"**Fasting BS > 120:** {input_data['fbs']}")
with col3:
    st.info(f"**Max Heart Rate:** {input_data['thalch']}")
    st.info(f"**Exercise Angina:** {input_data['exang']}")
    st.info(f"**Oldpeak:** {input_data['oldpeak']}")
    st.info(f"**Resting ECG:** {input_data['restecg']}")


# --- Prediction Logic ---
# This block is executed when the user clicks the 'Predict' button.
if st.sidebar.button('Predict Heart Disease'):
    # Define the URL of your backend API
    # This should match the address where your Flask app is running.
    API_URL = "http://127.0.0.1:5000/predict"

    with st.spinner('Sending data to the model and awaiting prediction...'):
        try:
            # Send a POST request to the backend with the user's data as JSON.
            response = requests.post(API_URL, json=input_data)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            # Get the prediction result from the JSON response.
            result = response.json()
            prediction = result['prediction']
            probability = result['probability']

            time.sleep(1) # A small delay for better user experience

            # Display the prediction result
            st.subheader("Prediction Result from the Model")
            if prediction == "Has Disease":
                st.error(f"**Result: High Risk of Heart Disease**")
                st.metric(label="Confidence (Probability of Disease)", value=f"{probability}%")
                st.progress(probability / 100.0)
                st.warning("This patient shows a high likelihood of having heart disease. Further medical consultation is strongly advised.")
            else:
                st.success(f"**Result: Low Risk of Heart Disease**")
                st.metric(label="Confidence (Probability of Disease)", value=f"{probability}%")
                st.progress(probability / 100.0)
                st.info("This patient shows a low likelihood of having heart disease based on the provided data.")

        except requests.exceptions.RequestException as e:
            # Handle connection errors or other request issues
            st.error(f"Could not connect to the backend server. Please ensure it is running.")
            st.error(f"Error details: {e}")

st.markdown("---")
st.markdown("*Disclaimer: This prediction is based on a machine learning model and is not a substitute for professional medical advice.*")
