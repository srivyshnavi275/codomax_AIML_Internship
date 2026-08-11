import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load dataset
df = pd.read_csv("student_scores_cleaned.csv")

print("DAY 7 - MACHINE LEARNING BASICS")
print("=" * 50)

# 2. Create target: Average Score
score_cols = [
    "math_score",
    "physics_score",
    "chemistry_score",
    "english_score",
    "cs_score"
]

df["avg_score"] = df[score_cols].mean(axis=1)

# 3. Select features and target
features = [
    "math_score",
    "physics_score",
    "chemistry_score",
    "english_score",
    "cs_score",
    "attendance_pct",
    "study_hours_per_week",
    "age"
]

X = df[features]
y = df["avg_score"]

print("\nFeatures:", features)
print("Target: avg_score")

# 4. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 5. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-" * 30)
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"R²   : {r2:.2f}")
# 8. Predict a new student's score
new_student = [[85, 80, 78, 75, 82, 90, 15, 20]]
prediction = model.predict(new_student)
print("\nNEW STUDENT PREDICTION")
print("-" * 30)
print(f"Predicted Average Score: {prediction[0]:.2f}")