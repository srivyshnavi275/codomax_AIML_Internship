import numpy as np
from sklearn.linear_model import LinearRegression
study_hours = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,
    0.5, 1.2, 2.8, 3.2, 4.8, 5.2, 6.8, 7.2, 8.8, 9.2
]).reshape(-1, 1)
scores = np.array([
    25, 35, 45, 52, 60, 68, 72, 78, 85, 90,
    30, 40, 50, 58, 65, 70, 75, 80, 88,
    20, 28, 42, 48, 62, 66, 73, 77, 86, 89
])
# Train model
model = LinearRegression()
model.fit(study_hours, scores)
print("=" * 40)
print("STUDY HOURS -> SCORE PREDICTION")
print("=" * 40)
while True:
    try:
        hours = input("Enter study hours (q to quit): ")
        if hours.lower() == "q":
            print("Goodbye!")
            break
        hours = float(hours)
        if hours < 0 or hours > 24:
            print("Enter hours between 0 and 24.")
            continue
        # Predict score
        score = model.predict([[hours]])[0]
        score = max(0, min(100, round(score, 2)))
        print(f"Predicted Score: {score}%")
        if score >= 90:
            print("Excellent!")
        elif score >= 75:
            print("Good preparation!")
        elif score >= 60:
            print("Decent. Study a little more.")
        else:
            print("Try studying more.")

    except ValueError:
        print("Please enter a valid number.")