import pandas as pd
import numpy as np
import pickle
import streamlit as st

# Load the saved model
diabetes_model = pickle.load(open('random_forest_data.pkl', 'rb'))

# Application Title
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("🩺 Diabetes Prediction Web App")
st.write("This web app predicts whether a woman aged 21+ is diabetic based on health parameters.")

# Adding a node
st.markdown(
     """
    🔍 **Note:** This tool is designed for use by **women aged 21 and above**.
    The prediction is based on a rigorously trained machine learning model using the PIMA Indian dataset.
    """,
    unsafe_allow_html=True
)

# Sidebar for Input

st.sidebar.header("Patient data Input")

def user_input():
    pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
    glucose = st.sidebar.slider("Glucose", 0, 200, 120)
    blood_pressure=  st.sidebar.slider("Blood Pressure", 0, 140, 70)
    skin_thickness= st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0, 900, 80)
    bmi= st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0,  3.0, 0.5)
    age = st.sidebar.slider("Age", 21,100,30)
  
  data = {'Pregnancies': pregnancies, 'Glucose' : glucose, 'BloodPressure': blood_pressure, 'SkinThickness': skin_thickness, 'Insulin': insulin, 'BMI': bmi, 'DiabetesPedigreeFunction': dpf, 'Age': age}
  return pd.DataFrame([data])

input_df = user_input()

st.subheader("Entered Data:")
st.write(input_df)

if st.button("Predict"):
  prediction = diabetes_model.predict(input_df)[0]
  result = "Diabetic" if  prediction == 1 else  "Non-Diabetic"
  st.subheader("Prediction: ")
  st.success(f"The patient is **{result}**.")