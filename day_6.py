
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots folder
os.makedirs("plots", exist_ok=True)

# Load dataset
file_path = r"C:\Users\my pc\Downloads\student_scores_cleaned.csv"
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Rows and columns:", df.shape)

# Score columns
scores = [
    "math_score",
    "physics_score",
    "chemistry_score",
    "english_score",
    "cs_score"
]

# 1. Scatter Plot - Math vs Physics
sns.scatterplot(
    data=df,
    x="math_score",
    y="physics_score",
    hue="placement_status"
)
plt.title("Math vs Physics")
plt.savefig("plots/scatter.png")
plt.close()


# 2. Bar Chart - Average Scores by Department
df.groupby("department")[scores].mean().plot(kind="bar")

plt.title("Average Scores by Department")
plt.ylabel("Average Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/bar_department.png")
plt.close()


# 3. Bar Chart - Average Math Score by Activity
df.groupby("extracurricular")["math_score"].mean().plot(kind="bar")

plt.title("Math Score by Extracurricular Activity")
plt.ylabel("Average Math Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/bar_activity.png")
plt.close()


# 4. Line Chart - Average Score by Age
df["average_score"] = df[scores].mean(axis=1)

df.groupby("age")["average_score"].mean().plot(marker="o")

plt.title("Average Score by Age")
plt.xlabel("Age")
plt.ylabel("Average Score")
plt.grid()
plt.savefig("plots/line_age.png")
plt.close()


# 5. Heatmap - Correlation
corr = df[
    scores + [
        "attendance_pct",
        "study_hours_per_week",
        "age"
    ]
].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/heatmap.png")
plt.close()


# 6. Box Plot - Scores by Placement
data = df.melt(
    id_vars="placement_status",
    value_vars=scores,
    var_name="Subject",
    value_name="Score"
)

sns.boxplot(
    data=data,
    x="Subject",
    y="Score",
    hue="placement_status"
)

plt.title("Scores by Placement Status")
plt.tight_layout()
plt.savefig("plots/boxplot.png")
plt.close()


print("All plots created successfully!")
