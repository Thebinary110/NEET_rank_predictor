import streamlit as st
import joblib
import pandas as pd

rf_model = joblib.load(r"C:\Users\Dell\Desktop\company work\random_forest_model.pkl")
scaler = joblib.load(r"C:\Users\Dell\Desktop\company work\scaler.pkl")

st.title("NEET 2024 Marks vs Rank Predictor")

marks = st.number_input("Enter your marks:", min_value=0, max_value=720, value=500)
difficulty = st.selectbox("Select paper difficulty:", ["easy", "medium", "hard"])
model_choice = st.selectbox("Select model:", ["Random Forest", "Gradient Boosting"])

if st.button("Predict Rank"):
    scaled_marks = scaler.transform([[marks]])

    input_data = pd.DataFrame({
        'Marks': [scaled_marks[0][0]],
        'Difficulty': [difficulty]
    })
    input_data['Difficulty'] = input_data['Difficulty'].map({'easy': 0, 'medium': 1, 'hard': 2})

    if model_choice == "Random Forest":
        prediction = rf_model.predict(input_data)

    st.success(f"Predicted Rank: {int(prediction[0])}")