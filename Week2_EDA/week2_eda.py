import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv("Week1_Data_Cleaning/data/cleaned_train.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Dataset information
print("\nDataset Information:")
df.info()

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Survival count
print("\nSurvival Count:")
print(df["Survived"].value_counts())

# Survival rate by gender
print("\nSurvival Rate by Gender:")
print(df.groupby("Sex")["Survived"].mean())

# Survival rate by passenger class
print("\nSurvival Rate by Passenger Class:")
print(df.groupby("Pclass")["Survived"].mean())


# Visualization 1: Survival Count
plt.figure(figsize=(7, 5))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()


# Visualization 2: Survival by Gender
plt.figure(figsize=(7, 5))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.show()


# Visualization 3: Survival by Passenger Class
plt.figure(figsize=(7, 5))
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.show()


# Visualization 4: Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# Visualization 5: Fare Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Fare"], bins=20, kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()


# Visualization 6: Correlation Heatmap
plt.figure(figsize=(9, 6))
numeric_df = df.select_dtypes(include="number")
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# EDA Insights

print("\nEDA Insights:")

female_survival = df[df["Sex"] == "female"]["Survived"].mean() * 100
male_survival = df[df["Sex"] == "male"]["Survived"].mean() * 100

print("Female survival rate:", round(female_survival, 2), "%")
print("Male survival rate:", round(male_survival, 2), "%")

class_survival = df.groupby("Pclass")["Survived"].mean() * 100

print("\nSurvival Rate by Class:")
print(class_survival.round(2))

print("\nMost common passenger class:", df["Pclass"].mode()[0])
print("Average age:", round(df["Age"].mean(), 2))
print("Average fare:", round(df["Fare"].mean(), 2))