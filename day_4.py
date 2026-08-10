import pandas as pd

# Load the student score dataset
df = pd.read_csv("student_scores.csv")

# Display the first 5 rows
print("First 5 rows:")
print(df.head())

# Display the last 5 rows
print("\nLast 5 rows:")
print(df.tail())

# Display number of rows and columns
print("\nDataset shape (rows, columns):")
print(df.shape)

# Display column names
print("\nColumn names:")
print(df.columns)

# Display dataset information
print("\nDataset information:")
df.info()