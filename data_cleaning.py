import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\my pc\Downloads\student_scores.csv")

print(df.head())
print(df.isnull().sum())

# Replace None with NaN
df.replace("None", np.nan, inplace=True)

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Remove duplicate student IDs
if "student_id" in df.columns:
    df.drop_duplicates("student_id", inplace=True)

# Save
df.to_csv("student_scores_cleaned.csv", index=False)

print("Cleaning completed!")
print(df.isnull().sum())