
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# 1. Load dataset
df = pd.read_csv(r"C:\Users\my pc\Downloads\student_scores.csv")

print("Dataset shape:", df.shape)

# 2. Create target variable
score_cols = [
    "math_score",
    "physics_score",
    "chemistry_score",
    "english_score",
    "cs_score"
]

df["avg_score"] = df[score_cols].mean(axis=1)

# 3. Select features and target
features = score_cols + [
    "attendance_pct",
    "study_hours_per_week",
    "age"
]

X = df[features]
y = df["avg_score"]

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# 5. Create and train model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

print("Model trained successfully!")

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("MSE :", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("MAE :", round(mae, 2))
print("R2  :", round(r2, 2))

# 8. Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

print("\nCross Validation R2:")
print(cv_scores)
print("Average CV R2:", round(cv_scores.mean(), 2))

# 9. Save model
joblib.dump(model, "models/linear_regression_model.pkl")

print("\nModel saved successfully!")

# 10. Predict a new student
new_student = pd.DataFrame({
    "math_score": [85],
    "physics_score": [80],
    "chemistry_score": [78],
    "english_score": [75],
    "cs_score": [82],
    "attendance_pct": [90],
    "study_hours_per_week": [15],
    "age": [20]
})

prediction = model.predict(new_student)

print("\nPredicted Average Score:",
      round(prediction[0], 2))

# 11. Plot Actual vs Predicted
plt.figure(figsize=(7, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted Scores")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)

plt.tight_layout()
plt.savefig("plots/model_evaluation.png")
plt.show()

print("\nDay 8 completed successfully!")