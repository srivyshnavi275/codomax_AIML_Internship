# Day 9 - Student Score Prediction

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

print("DAY 9 - STUDENT SCORE PREDICTION")
print("=" * 40)

# 1. Load trained model
model = joblib.load("C:\\Users\\my pc\\Downloads\\linear_regression_model (1).pkl")

with open("C:\\Users\\my pc\\Downloads\\model_metadata.json") as f:
    metadata = json.load(f)

print("Model loaded successfully!")
print("Model:", metadata['model_type'])
print("R² Score:", round(metadata['test_r2'], 4))

# 2. Load cleaned dataset
df = pd.read_csv("C:\\Users\\my pc\\Downloads\\student_scores.csv")

# Calculate average score
score_cols = [
    'math_score', 'physics_score', 'chemistry_score',
    'english_score', 'cs_score'
]

df['avg_score'] = df[score_cols].mean(axis=1)

features = metadata['features']

# 3. Predict scores for different study hours
avg_values = df[features].mean()

study_hours = np.arange(0, 31, 5)
predictions = []

for hours in study_hours:
    student = avg_values.copy()
    student['study_hours_per_week'] = hours

    X = pd.DataFrame([student])[features]
    prediction = model.predict(X)[0]

    predictions.append(round(prediction, 2))

# Display predictions
result = pd.DataFrame({
    'Study Hours': study_hours,
    'Predicted Score': predictions
})

print("\nPredictions:")
print(result)

# Save predictions
result.to_csv('predictions_by_study_hours.csv', index=False)

# 4. Predict scores for existing students
X = df[features]
df['predicted_score'] = model.predict(X)

print("\nFirst 5 Student Predictions:")
print(df[['avg_score', 'predicted_score']].head())

df[['avg_score', 'predicted_score']].to_csv(
    'student_predictions.csv', index=False
)

# 5. Visualization
plt.figure(figsize=(8, 5))

plt.plot(
    study_hours,
    predictions,
    marker='o'
)

plt.xlabel('Study Hours per Week')
plt.ylabel('Predicted Average Score')
plt.title('Study Hours vs Predicted Score')
plt.grid(True)

plt.savefig('study_hours_prediction.png')
plt.show()

print("\nDay 9 Prediction Task Completed!")
print("Files created:")
print("- predictions_by_study_hours.csv")
print("- student_predictions.csv")
print("- study_hours_prediction.png")