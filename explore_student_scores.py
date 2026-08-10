"""
Explore a student score dataset with pandas.

Usage:
    Place your dataset (CSV) in the same folder as this script,
    update FILE_PATH below, then run:
        python explore_student_scores.py
"""

import pandas as pd

# ---- 1. Load the dataset ----
FILE_PATH = "student_scores.csv"  # change this to your actual file name/path

df = pd.read_csv(FILE_PATH)
print("Dataset loaded successfully\n")

# ---- 2. Explore rows ----
print("=" * 60)
print("First 5 rows:")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("Last 5 rows:")
print("=" * 60)
print(df.tail())

# ---- 3. Explore columns ----
print("\n" + "=" * 60)
print("Column names:")
print("=" * 60)
print(df.columns.tolist())

print("\n" + "=" * 60)
print("Shape (rows, columns):")
print("=" * 60)
print(df.shape)

# ---- 4. Dataset information ----
print("\n" + "=" * 60)
print("Dataset info (dtypes, non-null counts):")
print("=" * 60)
df.info()

print("\n" + "=" * 60)
print("Summary statistics (numeric columns):")
print("=" * 60)
print(df.describe())

print("\n" + "=" * 60)
print("Missing values per column:")
print("=" * 60)
print(df.isnull().sum())
