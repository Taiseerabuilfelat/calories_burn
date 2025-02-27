import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
model_filename = "calories_model.pkl"
with open(model_filename, "rb") as f:
    model = pickle.load(f)

# Streamlit app
def main():
    st.title("Calories Burned Prediction App")
    st.write("Enter the details below to predict calories burned.")
    
    # User input fields
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    duration = st.number_input("Exercise Duration (minutes)", min_value=1, max_value=300, value=30)
    heart_rate = st.number_input("Heart Rate (BPM)", min_value=50, max_value=200, value=120)
    body_temp = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0)
    
    # Convert gender to numeric
    gender_numeric = 1 if gender == "Male" else 0
    
    # Prediction
    if st.button("Predict Calories Burned"):
        input_data = np.array([[gender_numeric, age, height, weight, duration, heart_rate, body_temp]])
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated Calories Burned: {prediction:.2f}")

if __name__ == "__main__":
    main()
