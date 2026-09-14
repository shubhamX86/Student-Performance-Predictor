import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv("students.csv")


# ==========================================
# FEATURES AND TARGET
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
# CREATE MODELS
# ==========================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42
        ),

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


# ==========================================
# TRAIN AND COMPARE
# ==========================================

results = []

best_model = None
best_model_name = None
best_mae = float("inf")


for name, model in models.items():

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "R2 Score": r2
    })

    print("\nModel:", name)
    print("MAE:", round(mae, 2))
    print("R2 Score:", round(r2, 2))


    # Find best model
    if mae < best_mae:

        best_mae = mae
        best_model = model
        best_model_name = name


# ==========================================
# RESULTS
# ==========================================

print("\n================================")
print("MODEL COMPARISON")
print("================================")

for result in results:

    print(
        result["Model"],
        "| MAE:",
        round(result["MAE"], 2),
        "| R2:",
        round(result["R2 Score"], 2)
    )


# ==========================================
# SAVE BEST MODEL
# ==========================================

with open("student_model.pkl", "wb") as file:

    pickle.dump(best_model, file)


print("\n================================")
print("BEST MODEL")
print("================================")

print("Best Model:", best_model_name)
print("Best MAE:", round(best_mae, 2))

print("\nBest model saved as student_model.pkl")