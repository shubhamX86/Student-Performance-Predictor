# 🎓 Student Performance Predictor

A Machine Learning project that predicts a student's final exam score
based on study hours, attendance, previous exam performance, assignment
score, and internal marks.

The project also compares multiple Machine Learning regression
algorithms and selects the best-performing model using Mean Absolute
Error (MAE).

------------------------------------------------------------------------

## 📌 Project Overview

The **Student Performance Predictor** is designed to demonstrate a
complete Machine Learning workflow:

``` text
Dataset
   ↓
Data Loading
   ↓
Data Preparation
   ↓
Train/Test Split
   ↓
Multiple ML Models
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Prediction
   ↓
Streamlit Dashboard
```

The application allows a user to enter student information and receive:

-   Predicted final score
-   Performance category
-   Student risk indication
-   Model comparison
-   Dataset analysis graphs

------------------------------------------------------------------------

# 1. 🎯 Project Objective

The main objectives of this project are:

-   Predict a student's final exam score.
-   Understand how different student factors affect performance.
-   Compare different Machine Learning regression algorithms.
-   Select the best model using evaluation metrics.
-   Build an interactive web application using Streamlit.
-   Provide a simple and understandable student performance analysis.

------------------------------------------------------------------------

# 2. 🛠️ Technologies Used

### Programming Language

-   Python

### Libraries

-   Pandas --- data handling and analysis
-   NumPy --- numerical operations
-   Scikit-learn --- Machine Learning
-   Matplotlib --- data visualization
-   Streamlit --- web application/dashboard
-   Pickle --- saving and loading the trained model

### Development Tools

-   VS Code
-   PowerShell / Command Prompt
-   Python Virtual Environment (`.venv`)

------------------------------------------------------------------------

# 3. 📂 Project Structure

``` text
Student-Performance-Predictor/
│
├── .venv/
│
├── .devcontainer/
│
├── students.csv
├── train.py
├── student_model.pkl
├── app.py
└── README.md
```

### File Description

  File                  Purpose
  --------------------- -------------------------------
  `students.csv`        Student dataset
  `train.py`            Trains and compares ML models
  `student_model.pkl`   Stores the best trained model
  `app.py`              Streamlit web application
  `README.md`           Project documentation

------------------------------------------------------------------------

# 4. 📊 Dataset

The project uses a CSV dataset containing student performance
information.

### Features

  Feature              Description
  -------------------- -------------------------------
  `study_hours`        Daily study hours
  `attendance`         Student attendance percentage
  `previous_score`     Previous exam score
  `assignment_score`   Assignment score
  `internal_marks`     Internal examination marks
  `final_score`        Final exam score / target

### Example

``` csv
study_hours,attendance,previous_score,assignment_score,internal_marks,final_score
2,60,55,60,58,56
3,70,60,65,62,63
4,75,65,70,68,69
5,80,70,75,72,75
6,85,75,80,78,81
```

`final_score` is the **target variable** that the Machine Learning model
predicts.

> **Note:** The current dataset is a small demo dataset created for
> learning and project demonstration. A larger real-world dataset should
> be used for production-level conclusions.

------------------------------------------------------------------------

# 5. 💻 Installation

Make sure Python is installed.

Check Python:

``` powershell
python --version
```

Install the required libraries:

``` powershell
python -m pip install pandas numpy scikit-learn matplotlib streamlit
```

Verify Streamlit:

``` powershell
python -m streamlit --version
```

------------------------------------------------------------------------

# 6. 📁 Create the Project

Create a project folder:

``` text
Student-Performance-Predictor
```

Open the folder in VS Code.

Create the following files:

``` text
students.csv
train.py
app.py
README.md
```

The dataset should be placed inside the project folder.

------------------------------------------------------------------------

# 7. 🤖 Train the Machine Learning Model

The `train.py` program:

1.  Loads the dataset.
2.  Selects input features.
3.  Separates features and target.
4.  Splits the dataset into training and testing data.
5.  Trains multiple regression models.
6.  Calculates MAE and R² Score.
7.  Finds the best model.
8.  Saves the best model as `student_model.pkl`.

Run:

``` powershell
python train.py
```

The trained model is saved as:

``` text
student_model.pkl
```

------------------------------------------------------------------------

# 8. 🌐 Run the Streamlit Application

Start the application using:

``` powershell
python -m streamlit run app.py
```

After starting the application, open:

``` text
http://localhost:8501
```

The application provides input fields for:

-   Daily Study Hours
-   Attendance
-   Previous Exam Score
-   Assignment Score
-   Internal Marks

After clicking:

``` text
🔮 Predict Performance
```

the application predicts the final score.

------------------------------------------------------------------------

# 9. 🧪 Model Comparison

The project compares four Machine Learning regression algorithms:

### 1. Linear Regression

A basic regression algorithm that models the relationship between input
features and the target score.

### 2. Decision Tree Regressor

Uses decision rules to predict the final score.

### 3. Random Forest Regressor

Uses multiple decision trees and combines their predictions.

### 4. Gradient Boosting Regressor

Builds models sequentially to improve prediction performance.

------------------------------------------------------------------------

## Evaluation Metrics

### Mean Absolute Error (MAE)

MAE represents the average absolute difference between actual and
predicted values.

**Lower MAE is better.**

For example:

``` text
MAE = 0.59
```

means that the model's predictions differed from the actual values by
approximately 0.59 marks on average on the evaluated test data.

### R² Score

R² indicates how well the model explains the variation in the target
variable.

A value closer to 1 generally indicates a better fit.

------------------------------------------------------------------------

# 10. 🏆 Best Model and Dashboard

For the current demo dataset and test split, the best model obtained
was:

``` text
Best Model: Linear Regression
Best MAE: 0.59
```

The application displays model comparison results and provides visual
analysis such as:

-   Study Hours vs Final Score
-   Attendance vs Final Score
-   Previous Score vs Final Score
-   Student dataset table
-   Model MAE
-   Model R² Score

The application also categorizes predicted performance:

    Predicted Score Performance
  ----------------- --------------
            80--100 Excellent 🏆
             60--79 Good 👍
             40--59 Average 🙂
              0--39 Poor ⚠️

------------------------------------------------------------------------

# 🔄 Complete Working Process

``` text
                    STUDENT DATA
                         │
                         ▼
                  students.csv
                         │
                         ▼
                  Data Preparation
                         │
                         ▼
                  Train/Test Split
                         │
                         ▼
              ┌─────────────────────┐
              │   Machine Learning  │
              │       Models        │
              └──────────┬──────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      Linear         Decision        Random
     Regression        Tree           Forest
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
                  Gradient Boosting
                         │
                         ▼
                  Model Evaluation
                         │
                         ▼
                    Best Model
                         │
                         ▼
                 student_model.pkl
                         │
                         ▼
                   Streamlit App
                         │
                         ▼
                 Student Input
                         │
                         ▼
                Predicted Score
                         │
                         ▼
              Performance Category
```

------------------------------------------------------------------------

# 📸 Application Features

The Streamlit dashboard contains:

-   🎓 Student Performance Predictor
-   📝 Student input form
-   🔮 Final score prediction
-   🏆 Performance classification
-   ⚠️ Risk indication
-   🤖 Model comparison
-   📊 MAE and R² Score
-   📈 Performance graphs
-   📋 Dataset visualization

------------------------------------------------------------------------

# 🧠 Machine Learning Concepts Demonstrated

This project covers the following ML concepts:

-   Dataset creation
-   Data loading
-   Feature selection
-   Target variable
-   Train/Test Split
-   Supervised Learning
-   Regression
-   Model Training
-   Model Prediction
-   MAE
-   R² Score
-   Model Comparison
-   Best Model Selection
-   Model Serialization using Pickle
-   Data Visualization
-   Web Deployment with Streamlit

------------------------------------------------------------------------

# ▶️ How to Run the Project

Open PowerShell inside the project directory:

``` powershell
cd E:\Project\Student-Performance-Predictor
```

Install dependencies:

``` powershell
python -m pip install pandas numpy scikit-learn matplotlib streamlit
```

Train the models:

``` powershell
python train.py
```

Run the application:

``` powershell
python -m streamlit run app.py
```

Open:

``` text
http://localhost:8501
```

------------------------------------------------------------------------

# ⚠️ Limitations

The current project has a small demonstration dataset.

Therefore:

-   Model evaluation can change significantly with a larger dataset.
-   The current MAE of 0.59 should not be interpreted as
    production-level accuracy.
-   More student records are required for reliable real-world
    predictions.
-   Additional factors could improve the prediction model.

------------------------------------------------------------------------

# 🚀 Future Improvements

Possible future improvements include:

-   Use a larger real-world student dataset.
-   Add more student features.
-   Add feature importance analysis.
-   Add prediction history.
-   Add student recommendations.
-   Add login/authentication.
-   Add downloadable prediction reports.
-   Add interactive dashboards.
-   Deploy the application online.
-   Compare additional ML algorithms.
-   Add cross-validation and hyperparameter tuning.

------------------------------------------------------------------------

# 👨‍🎓 Project Type

**Artificial Intelligence / Machine Learning**

**Domain:** Education

**Task:** Student Performance Prediction

**Learning Type:** Supervised Learning

**Problem Type:** Regression

**Best Model on Current Demo Dataset:** Linear Regression

**Current Best MAE:** 0.59

------------------------------------------------------------------------

# 📜 Conclusion

The Student Performance Predictor demonstrates how Machine Learning can
be used to predict student academic performance.

The project follows a complete workflow from dataset preparation and
model training to model evaluation and interactive prediction through a
Streamlit web application.

The current implementation successfully compares multiple regression
algorithms and selects the best-performing model based on Mean Absolute
Error.
