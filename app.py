import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================
# PAGE
# ==========================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)


# ==========================================
# LOAD DATA
# ==========================================

data = pd.read_csv("students.csv")


# ==========================================
# FEATURES
# ==========================================

features = [
    "study_hours",
    "attendance",
    "previous_score",
    "assignment_score",
    "internal_marks"
]

X = data[features]
y = data["final_score"]


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# LOAD BEST MODEL
# ==========================================

with open("student_model.pkl", "rb") as file:
    best_model = pickle.load(file)


# ==========================================
# MODEL COMPARISON
# ==========================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(random_state=42),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            random_state=42
        )
}


results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        prediction
    )

    r2 = r2_score(
        y_test,
        prediction
    )

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "R² Score": round(r2, 2)
    })


results_df = pd.DataFrame(results)


# ==========================================
# TITLE
# ==========================================

st.title("🎓 Student Performance Predictor")

st.write(
    "Predict student final exam performance using Machine Learning."
)


# ==========================================
# MODEL PERFORMANCE
# ==========================================

st.header("🤖 Model Comparison")

st.dataframe(
    results_df,
    use_container_width=True
)


# ==========================================
# BEST MODEL
# ==========================================

best_row = results_df.loc[
    results_df["MAE"].idxmin()
]

st.success(
    f"🏆 Best Model: {best_row['Model']} "
    f"| MAE: {best_row['MAE']}"
)


# ==========================================
# INPUT
# ==========================================

st.header("📝 Enter Student Information")

col1, col2 = st.columns(2)


with col1:

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


with col2:

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


# ==========================================
# PREDICTION
# ==========================================

if st.button("🔮 Predict Performance"):

    input_data = pd.DataFrame(
        [[
            study_hours,
            attendance,
            previous_score,
            assignment_score,
            internal_marks
        ]],
        columns=features
    )

    prediction = best_model.predict(
        input_data
    )[0]

    prediction = max(
        0,
        min(100, prediction)
    )


    st.header("📊 Prediction Result")

    st.success(
        f"Predicted Final Score: {prediction:.2f}"
    )


    # Performance

    if prediction >= 80:

        performance = "Excellent 🏆"

    elif prediction >= 60:

        performance = "Good 👍"

    elif prediction >= 40:

        performance = "Average 🙂"

    else:

        performance = "Poor ⚠️"


    st.info(
        f"Performance Level: {performance}"
    )


    # Risk

    if prediction < 40:

        st.error(
            "⚠️ Student is at high risk."
        )

    elif prediction < 60:

        st.warning(
            "⚠️ Student needs improvement."
        )

    else:

        st.success(
            "✅ Student is performing well."
        )


# ==========================================
# GRAPHS
# ==========================================

st.divider()

st.header("📈 Student Data Analysis")


# Study Hours

st.subheader("Study Hours vs Final Score")

fig1, ax1 = plt.subplots()

ax1.scatter(
    data["study_hours"],
    data["final_score"]
)

ax1.set_xlabel("Study Hours")
ax1.set_ylabel("Final Score")
ax1.set_title("Study Hours vs Final Score")

st.pyplot(fig1)


# Attendance

st.subheader("Attendance vs Final Score")

fig2, ax2 = plt.subplots()

ax2.scatter(
    data["attendance"],
    data["final_score"]
)

ax2.set_xlabel("Attendance (%)")
ax2.set_ylabel("Final Score")
ax2.set_title("Attendance vs Final Score")

st.pyplot(fig2)


# Previous Score

st.subheader("Previous Score vs Final Score")

fig3, ax3 = plt.subplots()

ax3.scatter(
    data["previous_score"],
    data["final_score"]
)

ax3.set_xlabel("Previous Score")
ax3.set_ylabel("Final Score")
ax3.set_title("Previous Score vs Final Score")

st.pyplot(fig3)


# ==========================================
# DATASET
# ==========================================

st.header("📋 Student Dataset")

st.dataframe(
    data,
    use_container_width=True
)