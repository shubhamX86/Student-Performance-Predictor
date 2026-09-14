import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle


# Dataset load
data = pd.read_csv("students.csv")

# Input features
X = data[
    [
        "study_hours",
        "attendance",
        "previous_score",
        "assignment_score",
        "internal_marks"
    ]
]

# Target
y = data["final_score"]


# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# Training
model.fit(X_train, y_train)


# Prediction
predictions = model.predict(X_test)


# Evaluation
error = mean_absolute_error(y_test, predictions)

print("Model trained successfully!")
print("Mean Absolute Error:", error)


# Save model
with open("student_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")