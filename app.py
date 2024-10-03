import pandas as pd
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Instantiating the trained model
model = tf.keras.models.load_model('model_0.h5')

#For data Preprocessing
with open('label_encoder_gender.pkl','rb') as file:
    gender_encoder = pickle.load(file)

with open('label_encoder_smoking_history.pkl','rb') as file:
    smoking_encoder = pickle.load(file)

with open('standard_scaler.pkl', 'rb') as file:
    sc = pickle.load(file)

# Streamlitting
st.title('Diabetes Prediction')
import streamlit as st
import pandas as pd

# Input fields using Streamlit
gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])  # Gender select box
age = round(float(st.slider('Age', 10, 100)), 1)  # Age slider
hypertension = st.selectbox('Having hypertension?', [0, 1])  # Hypertension select box (0/1)
heart_disease = st.selectbox('Have heart disease?', [0, 1])  # Heart disease select box (0/1)
smoking_history = st.selectbox('Smoke?', ['ever', 'never', 'No info', 'not current', 'former', 'current'])  # Smoking history options
bmi = round(float(st.number_input('BMI')), 2)  # BMI input
HbA1c_level = round(float(st.number_input('HbA1c levels')), 1)  # HbA1c levels input
blood_glucose_level = st.number_input('Blood Glucose Level', min_value=0, max_value=500, value=140)  # Blood glucose level input with default 140

# Mapping gender to numerical values to match the DataFrame

# Creating a dictionary for the input data
input_data = {
    'gender': gender_encoder.transform([gender]),
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'smoking_history': smoking_encoder.transform([smoking_history]),
    'bmi': bmi,
    'HbA1c_level': HbA1c_level,
    'blood_glucose_level': blood_glucose_level
}

# Converting the dictionary to a DataFrame
input_df = pd.DataFrame([input_data])

# Display the resulting DataFrame
st.write(input_df)

input_df = sc.transform(input_df)

y_pred = np.round(model.predict(input_df))
if y_pred == 0:
    st.write("You do not have diabetes")
else:
    st.write("You have diabetes, get checked!")
