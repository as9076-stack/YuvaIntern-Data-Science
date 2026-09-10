import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load cleaned Titanic dataset
df = pd.read_csv("Week1_Data_Cleaning/data/cleaned_train.csv")

print("TITANIC SURVIVAL PREDICTION - CAPSTONE PROJECT")
print("=" * 50)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# Data quality check
print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())


# Exploratory Data Analysis
print("\nSurvival Count:")
print(df["Survived"].value_counts())

print("\nSurvival Rate:")
print((df["Survived"].mean() * 100).round(2), "%")

print("\nSurvival Rate by Gender:")
print((df.groupby("Sex")["Survived"].mean() * 100).round(2))

print("\nSurvival Rate by Passenger Class:")
print((df.groupby("Pclass")["Survived"].mean() * 100).round(2))


# EDA Visualization
plt.figure(figsize=(7, 5))

df["Survived"].value_counts().plot(kind="bar")

plt.title("Titanic Survival Count")
plt.xlabel("Survival")
plt.ylabel("Number of Passengers")
plt.xticks([0, 1], ["Not Survived", "Survived"], rotation=0)

plt.show()


# Select features and target
features = ["Age", "Fare", "Pclass", "SibSp", "Parch"]

X = df[features]
y = df["Survived"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# Standardization
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature standardization completed.")


# Supervised Learning Model
model = LogisticRegression(random_state=42)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("\nLOGISTIC REGRESSION RESULTS")
print("-" * 30)

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Feature importance using coefficients
coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_[0]
})

coefficients = coefficients.sort_values(
    by="Coefficient",
    ascending=False
)

print("\nFeature Coefficients:")
print(coefficients.round(4))


# Coefficient Visualization
plt.figure(figsize=(8, 5))

plt.bar(
    coefficients["Feature"],
    coefficients["Coefficient"]
)

plt.title("Feature Coefficients - Logistic Regression")
plt.xlabel("Features")
plt.ylabel("Coefficient")

plt.show()


# Actual vs Predicted
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nActual vs Predicted - First 20:")
print(comparison.head(20))


# Actual vs Predicted Visualization
plt.figure(figsize=(7, 5))

plt.hist(
    [y_test, y_pred],
    bins=[-0.5, 0.5, 1.5],
    label=["Actual", "Predicted"],
    rwidth=0.8
)

plt.title("Actual vs Predicted Survival")
plt.xlabel("Survival Class")
plt.ylabel("Count")

plt.xticks([0, 1], ["Not Survived", "Survived"])

plt.legend()

plt.show()


# Final Summary
print("\nFINAL CAPSTONE SUMMARY")
print("=" * 50)

print("Dataset:", "Titanic")
print("Number of Rows:", len(df))
print("Number of Features Used:", len(features))
print("Model:", "Logistic Regression")
print("Test Accuracy:", round(accuracy * 100, 2), "%")

print("\nCapstone project completed successfully!")