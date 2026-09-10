import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the cleaned dataset
df = pd.read_csv("Week1_Data_Cleaning/data/cleaned_train.csv")

# Select features and target
features = ["Age", "Fare", "Pclass", "SibSp", "Parch"]

X = df[features]
y = df["Survived"]

print("Selected Features:")
print(X.head())

print("\nTarget:")
print(y.head())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Standardize the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nData standardized successfully!")

# Create Logistic Regression model
model = LogisticRegression(random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

print("\nModel training completed!")

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

# Confusion Matrix Visualization

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.colorbar()

plt.xticks([0, 1], ["Not Survived", "Survived"])
plt.yticks([0, 1], ["Not Survived", "Survived"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()

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

# Feature Coefficients

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