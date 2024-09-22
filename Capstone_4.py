
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load models and scalers with caching
@st.cache_data
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

@st.cache_data
def load_encoders_and_scalers():
    encoders = {}
    scalers = {}
    
    # Categorical columns that require encoding
    categorical_columns = [
        'item_date', 'customer', 'country', 'item type', 
        'application', 'material_ref', 'product_ref', 
        'delivery date'
    ]
    
    # Numerical columns that only require scaling
    numerical_columns = ['quantity tons', 'width', 'thickness', 'area']
    
    # Load encoders for categorical columns
    for col in categorical_columns:
        encoders[col] = load_pickle(f'label_encoder_{col}.pkl')
        scalers[col] = load_pickle(f'min_max_scaler_{col}.pkl')
    
    # Load scalers for numerical columns
    for col in numerical_columns:
        scalers[col] = load_pickle(f'min_max_scaler_{col}.pkl')
    
    return encoders, scalers

# Cache the loading of models and unique values
xgb_classifier = load_pickle('xgb_classifier_model.pkl')
etr_regressor = load_pickle('etr_regressor_model.pkl')
unique_values = load_pickle('unique_values.pkl')

# Load the encoders and scalers
encoders, scalers = load_encoders_and_scalers()

st.set_page_config(page_title="Industrial Copper Price Prediction", page_icon=":bar_chart:",layout="wide")

st.title("Industrial Copper Price Prediction")
st.header("Please fill the below details", divider=True)

hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)

def log_transform(value):
    return np.log(value) if value > 0 else 0

def preprocess_input(data, encoders, scalers):
    for key in data:
        if key in encoders:
            # Encode categorical variables
            data[key] = encoders[key].transform([str(data[key])])[0]
            # Scale the encoded values
            data[key] = scalers[key].transform([[data[key]]])[0][0]
        elif key in scalers:
            # Apply log transformation for specific numerical columns if needed
            if key in ['quantity tons', 'thickness', 'area']:
                data[key] = log_transform(data[key])
            # Scale numerical variables
            data[key] = scalers[key].transform([[data[key]]])[0][0]
    return data

with st.form("my_form"):
    # Capture the original input values
    inputs = {
        'item_date': str(st.selectbox('Select item_date in YYYYMMDD', unique_values['item_date'],index=None, placeholder="Select the date")),
        'customer': str(st.selectbox('Select customer', unique_values['customer'],index=None, placeholder="Select the customer")),
        'country': str(st.selectbox('Select country', unique_values['country'],index=None, placeholder="Select the country")),
        'item type': str(st.selectbox('Select item type', unique_values['item type'],index=None, placeholder="Select the type")),
        'application': str(st.selectbox('Select application', unique_values['application'],index=None, placeholder="Select the application")),
        'material_ref': str(st.selectbox('Select material_ref', unique_values['material_ref'], index=None, placeholder="Select the material reference")),
        'product_ref': str(st.selectbox('Select product_ref', unique_values['product_ref'], index=None, placeholder="Select the product reference")),
        'delivery date': str(st.selectbox('Select delivery date in YYYYMMDD', unique_values['delivery date'],index=None, placeholder="Select the date")),
        'quantity tons': st.number_input('Enter quantity tons', value=0.0, min_value=0.0),
        'thickness': st.number_input('Enter thickness', value=0.0, min_value=0.0),
        'width': st.number_input('Enter width', value=0.0, min_value=0.0)
    }
    
    inputs['area'] = inputs['thickness'] * inputs['width']

    # Store the original input values in a separate DataFrame, with selectbox values as strings
    original_inputs_df = pd.DataFrame([inputs])

    predict_button = st.form_submit_button("Predict")
    clear_button = st.form_submit_button("Clear")

    if predict_button:
        # Preprocess the input for model prediction
        scaled_inputs = preprocess_input(inputs, encoders, scalers)

        # Ensure the correct order of columns for both original and scaled DataFrames
        column_order = [
            'item_date', 'customer', 'country', 'item type', 
            'application', 'material_ref', 'product_ref', 
            'delivery date', 'quantity tons', 'width', 
            'thickness', 'area'
        ]

        # Create DataFrame with scaled values for prediction
        df = pd.DataFrame([scaled_inputs], columns=column_order)

        # Ensure the original input DataFrame has the correct column order
        original_inputs_df = original_inputs_df[column_order]

        # Make predictions
        predictions_class = xgb_classifier.predict(df)
        predictions_reg = etr_regressor.predict(df)
        
        label_mapping = {0: 'Lost', 1: 'Won'}
        predicted_label = label_mapping[predictions_class[0]]

        # Display the original input values DataFrame with selectbox values as strings
        st.write("Input DataFrame (Original Values):")
        st.write(original_inputs_df)

        status_color = "green" if predicted_label == "Won" else "red"
        st.markdown(f'<span style="color:{status_color}; font-weight:bold;">Predicted Status: {predicted_label} {"👍" if predicted_label == "Won" else "👎"}</span>', unsafe_allow_html=True)
        st.markdown(f'<span style="color:blue; font-weight:bold;">Predicted Selling Price: {predictions_reg[0]}</span>', unsafe_allow_html=True)


    if clear_button:
        st.experimental_rerun()
