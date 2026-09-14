import streamlit as st
import pickle
import numpy as np


# Load model
with open("student_model.pkl", "rb") as file:
    model = pickle.load(file)


# Page title
st.title("🎓 Student Performance Predictor")

st.write(
    "Enter the student's information to predict final performance."
)


# Inputs
study_hours = st.number_input(
    "Daily Study Hours",
    min_value=0.0,
    max_value=15.0,
    value=3.0
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

previous_score = st.number_input(
    "Previous Exam Score",
    min_value=0.0,
    max_value=100.0,
    value=60.0
)

assignment_score = st.number_input(
    "Assignment Score",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

internal_marks = st.number_input(
    "Internal Marks",
    min_value=0.0,
    max_value=100.0,
    value=65.0
)


# Prediction button
if st.button("Predict Performance"):

    input_data = np.array([
        [
            study_hours,
            attendance,
            previous_score,
            assignment_score,
            internal_marks
        ]
    ])

    prediction = model.predict(input_data)[0]

    # Keep score between 0 and 100
    prediction = max(0, min(100, prediction))

    st.success(
        f"Predicted Final Score: {prediction:.2f}"
    )


    # Performance category
    if prediction >= 80:
        performance = "Excellent 🏆"

    elif prediction >= 60:
        performance = "Good 👍"

    elif prediction >= 40:
        performance = "Average 🙂"

    else:
        performance = "Poor ⚠️"


    st.info(f"Performance: {performance}")