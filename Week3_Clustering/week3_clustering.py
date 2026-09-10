import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the cleaned dataset
df = pd.read_csv("Week1_Data_Cleaning/data/cleaned_train.csv")

# Select features for clustering
features = ["Age", "Fare", "Pclass", "SibSp", "Parch"]
X = df[features]

print("Selected Features:")
print(X.head())

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nData standardized successfully!")

# Elbow Method
inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker="o")
plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")


plt.show()

# Print inertia values
print("\nInertia Values:")

for k, value in zip(range(1, 11), inertia):
    print("K =", k, "Inertia =", round(value, 2))

# Apply K-Means Clustering

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster Labels:")
print(df["Cluster"].value_counts().sort_index())

print("\nFirst 10 rows with Cluster:")
print(df[["PassengerId", "Age", "Fare", "Pclass", "Cluster"]].head(10))

# Cluster Visualization

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Age"],
    df["Fare"],
    c=df["Cluster"],
    cmap="viridis",
    alpha=0.7
)

plt.title("K-Means Clustering of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.colorbar(label="Cluster")

plt.show()

# Cluster Summary

cluster_summary = df.groupby("Cluster")[features].mean()

print("\nCluster Summary:")
print(cluster_summary.round(2))

# Number of passengers in each cluster

print("\nNumber of passengers in each cluster:")
print(df["Cluster"].value_counts().sort_index())