

import pandas as pd
import numpy as np

# 1. Create Student Dataset
np.random.seed(42)

data = {
    "student_id": [f"STU{i:03d}" for i in range(1, 101)],
    "name": [f"Student_{i}" for i in range(1, 101)],
    "age": np.random.randint(17, 25, 100),
    "gender": np.random.choice(["Male", "Female", "Other"], 100),
    "department": np.random.choice(
        ["Computer Science", "Electronics", "Mechanical", "Civil", "Chemical"], 100
    ),
    "math_score": np.random.randint(40, 101, 100),
    "physics_score": np.random.randint(40, 101, 100),
    "chemistry_score": np.random.randint(40, 101, 100),
    "english_score": np.random.randint(40, 101, 100),
    "cs_score": np.random.randint(40, 101, 100)
}

df = pd.DataFrame(data)

# 2. Save Dataset
df.to_csv("student_scores.csv", index=False)

# 3. Load Dataset
df = pd.read_csv("student_scores.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

# 4. View Data
print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nColumns:")
print(df.columns.tolist())

# 5. Dataset Information
print("\nDataset Info:")
df.info()

# 6. Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# 7. Remove Duplicates
print("\nDuplicate Rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# 8. Statistics
print("\nStatistical Summary:")
print(df.describe())

# 9. Simple Analysis
print("\nAverage Math Score:", round(df["math_score"].mean(), 2))
print("Average CS Score:", round(df["cs_score"].mean(), 2))

print("\nStudents by Department:")
print(df["department"].value_counts())

# 10. Save Clean Dataset
df.to_csv("clean_student_scores.csv", index=False)

print("\n✅ Day 4 Task Completed Successfully!")
print("Clean dataset saved as clean_student_scores.csv")

