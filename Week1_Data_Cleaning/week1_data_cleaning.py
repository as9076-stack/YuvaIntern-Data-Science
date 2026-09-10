import pandas as pd

# Load the dataset
df = pd.read_csv("Week1_Data_Cleaning/data/train.csv")

# Display the first 5 rows
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Shape:")
print(df.shape)

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Handle missing values

# Fill missing Age values with the median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with the most frequent value
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin column because it contains too many missing values
df = df.drop("Cabin", axis=1)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Check for duplicate rows after cleaning
print("\nDuplicate Rows After Cleaning:")
print(df.duplicated().sum())

# Check the shape after cleaning
print("\nDataset Shape After Cleaning:")
print(df.shape)

# Display the cleaned dataset
print("\nCleaned Dataset:")
print(df.head())

# Save the cleaned dataset
df.to_csv("Week1_Data_Cleaning/data/cleaned_train.csv", index=False)

print("\nCleaned dataset saved successfully!")

# Save the cleaned dataset
df.to_csv("Week1_Data_Cleaning/data/cleaned_train.csv", index=False)

print("\nCleaned dataset saved successfully!")

# Data Validation

print("\nData Validation:")

print("Age minimum:", df["Age"].min())
print("Age maximum:", df["Age"].max())

print("Fare minimum:", df["Fare"].min())
print("Fare maximum:", df["Fare"].max())

print("Pclass values:", df["Pclass"].unique())
print("Survived values:", df["Survived"].unique())
print("Sex values:", df["Sex"].unique())
print("Embarked values:", df["Embarked"].unique())

print("\nFinal Missing Values:")
print(df.isnull().sum())

# Outlier Analysis using IQR

print("\nOutlier Analysis:")

# Age outliers
Q1_age = df["Age"].quantile(0.25)
Q3_age = df["Age"].quantile(0.75)
IQR_age = Q3_age - Q1_age

lower_age = Q1_age - 1.5 * IQR_age
upper_age = Q3_age + 1.5 * IQR_age

age_outliers = df[(df["Age"] < lower_age) | (df["Age"] > upper_age)]

print("Age lower limit:", lower_age)
print("Age upper limit:", upper_age)
print("Number of Age outliers:", len(age_outliers))

# Fare outliers
Q1_fare = df["Fare"].quantile(0.25)
Q3_fare = df["Fare"].quantile(0.75)
IQR_fare = Q3_fare - Q1_fare

lower_fare = Q1_fare - 1.5 * IQR_fare
upper_fare = Q3_fare + 1.5 * IQR_fare

fare_outliers = df[(df["Fare"] < lower_fare) | (df["Fare"] > upper_fare)]

print("Fare lower limit:", lower_fare)
print("Fare upper limit:", upper_fare)
print("Number of Fare outliers:", len(fare_outliers))

# Visualize Outliers

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.boxplot(df["Fare"])
plt.title("Fare Outlier Analysis")
plt.ylabel("Fare")
plt.show()