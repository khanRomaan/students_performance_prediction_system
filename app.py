import streamlit as st
import pickle
import numpy as np

st.title("🎓 Student Performance Prediction System")
st.write("Enter student details to predict the final grade")

# Load model and encoders
model = pickle.load(open("student_model.pkl", "rb"))
le1, le2, le3 = pickle.load(open("encoders.pkl", "rb"))

# Inputs
col1, col2 = st.columns(2)
with col1:
    Hours_Studied = st.number_input("Hours Studied", 1, 15, 8)
    Attendance = st.number_input("Attendance (%)", 60, 100, 85)
    Previous_Grade = st.number_input("Previous Grade", 0, 100, 70)
    Study_Time = st.number_input("Daily Study Time (hours)", 1, 10, 5)
    Sleep_Hours = st.number_input("Sleep Hours", 4, 10, 7)
with col2:
    Assignments_Submitted = st.number_input("Assignments Submitted", 5, 20, 10)
    Parental_Education = st.selectbox("Parental Education", le1.classes_)
    School_Type = st.selectbox("School Type", le3.classes_)
    Extracurricular = st.selectbox("Extracurricular", le2.classes_)

# Encode categorical values
Parental_Education_enc = le1.transform([Parental_Education])[0]
School_Type_enc = le3.transform([School_Type])[0]
Extracurricular_enc = le2.transform([Extracurricular])[0]

# Prediction
if st.button("🎯 Predict Performance"):
    features = np.array([[Hours_Studied, Attendance, Previous_Grade,
                          Parental_Education_enc, Extracurricular_enc,
                          Study_Time, Sleep_Hours, Assignments_Submitted,
                          School_Type_enc]])
    prediction = model.predict(features)[0]
    st.success(f"Predicted Final Grade: {prediction *10:.2f}")
    st.info("Pass ✅" if prediction >= 50 else "Fail ❌")
