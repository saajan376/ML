import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Mushroom Classifier",
    page_icon="🍄",
    layout="wide"
)

# --- App Header ---
st.title("🍄 Mushroom Classifier")
st.write(
    "Select the features of a mushroom from the dropdowns below to predict if it's **edible** or **poisonous**."
)
st.markdown("---")


# --- Feature Definitions ---
# This dictionary defines all the features and their possible values (categories).
# It's the same structure used in the previous HTML/JS frontend.
features = {
    'cap-shape': { 'name': 'Cap Shape', 'options': { 'b': 'Bell', 'c': 'Conical', 'x': 'Convex', 'f': 'Flat', 'k': 'Knobbed', 's': 'Sunken' } },
    'cap-surface': { 'name': 'Cap Surface', 'options': { 'f': 'Fibrous', 'g': 'Grooves', 'y': 'Scaly', 's': 'Smooth' } },
    'cap-color': { 'name': 'Cap Color', 'options': { 'n': 'Brown', 'b': 'Buff', 'c': 'Cinnamon', 'g': 'Gray', 'r': 'Green', 'p': 'Pink', 'u': 'Purple', 'e': 'Red', 'w': 'White', 'y': 'Yellow' } },
    'bruises': { 'name': 'Bruises', 'options': { 't': 'Yes', 'f': 'No' } },
    'odor': { 'name': 'Odor', 'options': { 'a': 'Almond', 'l': 'Anise', 'c': 'Creosote', 'y': 'Fishy', 'f': 'Foul', 'm': 'Musty', 'n': 'None', 'p': 'Pungent', 's': 'Spicy' } },
    'gill-attachment': { 'name': 'Gill Attachment', 'options': { 'a': 'Attached', 'd': 'Descending', 'f': 'Free', 'n': 'Notched' } },
    'gill-spacing': { 'name': 'Gill Spacing', 'options': { 'c': 'Close', 'w': 'Crowded', 'd': 'Distant' } },
    'gill-size': { 'name': 'Gill Size', 'options': { 'b': 'Broad', 'n': 'Narrow' } },
    'gill-color': { 'name': 'Gill Color', 'options': { 'k': 'Black', 'n': 'Brown', 'b': 'Buff', 'h': 'Chocolate', 'g': 'Gray', 'r': 'Green', 'o': 'Orange', 'p': 'Pink', 'u': 'Purple', 'e': 'Red', 'w': 'White', 'y': 'Yellow' } },
    'stalk-shape': { 'name': 'Stalk Shape', 'options': { 'e': 'Enlarging', 't': 'Tapering' } },
    'stalk-root': { 'name': 'Stalk Root', 'options': { 'b': 'Bulbous', 'c': 'Club', 'u': 'Cup', 'e': 'Equal', 'z': 'Rhizomorphs', 'r': 'Rooted', '?': 'Missing' } },
    'stalk-surface-above-ring': { 'name': 'Stalk Surface Above Ring', 'options': { 'f': 'Fibrous', 'y': 'Scaly', 'k': 'Silky', 's': 'Smooth' } },
    'stalk-surface-below-ring': { 'name': 'Stalk Surface Below Ring', 'options': { 'f': 'Fibrous', 'y': 'Scaly', 'k': 'Silky', 's': 'Smooth' } },
    'stalk-color-above-ring': { 'name': 'Stalk Color Above Ring', 'options': { 'n': 'Brown', 'b': 'Buff', 'c': 'Cinnamon', 'g': 'Gray', 'o': 'Orange', 'p': 'Pink', 'e': 'Red', 'w': 'White', 'y': 'Yellow' } },
    'stalk-color-below-ring': { 'name': 'Stalk Color Below Ring', 'options': { 'n': 'Brown', 'b': 'Buff', 'c': 'Cinnamon', 'g': 'Gray', 'o': 'Orange', 'p': 'Pink', 'e': 'Red', 'w': 'White', 'y': 'Yellow' } },
    'veil-type': { 'name': 'Veil Type', 'options': { 'p': 'Partial', 'u': 'Universal' } },
    'veil-color': { 'name': 'Veil Color', 'options': { 'n': 'Brown', 'o': 'Orange', 'w': 'White', 'y': 'Yellow' } },
    'ring-number': { 'name': 'Ring Number', 'options': { 'n': 'None', 'o': 'One', 't': 'Two' } },
    'ring-type': { 'name': 'Ring Type', 'options': { 'c': 'Cobwebby', 'e': 'Evanescent', 'f': 'Flaring', 'l': 'Large', 'n': 'None', 'p': 'Pendant', 's': 'Sheathing', 'z': 'Zone' } },
    'spore-print-color': { 'name': 'Spore Print Color', 'options': { 'k': 'Black', 'n': 'Brown', 'b': 'Buff', 'h': 'Chocolate', 'r': 'Green', 'o': 'Orange', 'u': 'Purple', 'w': 'White', 'y': 'Yellow' } },
    'population': { 'name': 'Population', 'options': { 'a': 'Abundant', 'c': 'Clustered', 'n': 'Numerous', 's': 'Scattered', 'v': 'Several', 'y': 'Solitary' } },
    'habitat': { 'name': 'Habitat', 'options': { 'g': 'Grasses', 'l': 'Leaves', 'm': 'Meadows', 'p': 'Paths', 'u': 'Urban', 'w': 'Waste', 'd': 'Woods' } }
}

# --- Input Form ---
st.header("Mushroom Features")

# Use columns for a cleaner layout
cols = st.columns(3)
user_inputs = {}
col_index = 0

# Create a selectbox for each feature
for key, feature in features.items():
    # This function formats the display text while keeping the key as the underlying value
    def format_func(option_key):
        return feature['options'][option_key]

    user_inputs[key] = cols[col_index].selectbox(
        label=feature['name'],
        options=list(feature['options'].keys()),
        format_func=format_func,
        key=key
    )
    col_index = (col_index + 1) % 3


# --- Prediction Logic ---
if st.button('Predict', type="primary", use_container_width=True):
    # The URL for the Flask backend API
    api_url = 'http://127.0.0.1:5000/predict'
    
    # Show a spinner while waiting for the prediction
    with st.spinner('Asking the model...'):
        try:
            # Send the user inputs to the backend API
            response = requests.post(api_url, json=user_inputs)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            
            result = response.json()
            
            # Display the result
            st.markdown("---")
            st.subheader("Prediction Result")
            
            prediction = result.get('prediction')
            confidence = float(result.get('confidence', 0))
            
            if prediction == 'Poisonous':
                st.error(f"**Result: {prediction}**")
            else:
                st.success(f"**Result: {prediction}**")
            
            st.metric(label="Confidence", value=f"{confidence:.0%}")

        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the prediction service. Please ensure the backend is running. Error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

