# ============================================================
# DIABETES PATIENT CLUSTERING USING K-MEANS CLUSTERING
# ============================================================

# -------------------------
# Import Libraries
# -------------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("diabetes.csv")

print("\nDIABETES DATASET\n")
print(df.head())

# -------------------------
# Dataset Information
# -------------------------

print("\nDataset Information\n")
print(df.info())

print("\nMissing Values\n")
print(df.isnull().sum())

# -------------------------
# Replace Invalid Zero Values
# -------------------------
# In this dataset, these columns cannot realistically be zero.
# Replace zeros with the column mean.

cols = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for col in cols:
    df[col] = df[col].replace(0, df[col].mean())

# -------------------------
# Features
# -------------------------
# Outcome is ignored because K-Means is an
# Unsupervised Learning algorithm.

X = df.drop("Outcome", axis=1)

# -------------------------
# Feature Scaling
# -------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ============================================================
# ELBOW METHOD
# ============================================================

wcss = []

for i in range(1,11):

    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)

plt.figure(figsize=(8,5))

plt.plot(
    range(1,11),
    wcss,
    marker="o"
)

plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")

plt.grid(True)

plt.show()

# ============================================================
# TRAIN K-MEANS MODEL
# ============================================================

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

# ============================================================
# PCA FOR VISUALIZATION
# ============================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters,
    cmap="viridis",
    s=60
)

plt.scatter(
    pca.transform(kmeans.cluster_centers_)[:,0],
    pca.transform(kmeans.cluster_centers_)[:,1],
    color="red",
    marker="X",
    s=250,
    label="Centroids"
)

plt.title("Diabetes Patient Clusters")

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.legend()

plt.grid(True)

plt.show()

# ============================================================
# CLUSTER SUMMARY
# ============================================================

print("\nPatients in each Cluster\n")

print(df["Cluster"].value_counts())

print("\nCluster Means\n")

print(df.groupby("Cluster").mean())

# ============================================================
# OUTCOME VS CLUSTER
# ============================================================

print("\nOutcome Distribution in each Cluster\n")

comparison = pd.crosstab(
    df["Cluster"],
    df["Outcome"]
)

print(comparison)

# ============================================================
# USER INPUT
# ============================================================

print("\n========== PATIENT DETAILS ==========\n")

preg = int(input("Pregnancies : "))

glucose = float(input("Glucose : "))

bp = float(input("Blood Pressure : "))

skin = float(input("Skin Thickness : "))

insulin = float(input("Insulin : "))

bmi = float(input("BMI : "))

dpf = float(input("Diabetes Pedigree Function : "))

age = int(input("Age : "))

new_patient = pd.DataFrame({

    "Pregnancies":[preg],

    "Glucose":[glucose],

    "BloodPressure":[bp],

    "SkinThickness":[skin],

    "Insulin":[insulin],

    "BMI":[bmi],

    "DiabetesPedigreeFunction":[dpf],

    "Age":[age]

})

# Scale Input

new_patient_scaled = scaler.transform(new_patient)

# Predict Cluster

cluster = kmeans.predict(new_patient_scaled)

print("\n==============================")

print("Predicted Cluster :", cluster[0])

print("==============================")

if cluster[0] == 0:
    print("This patient belongs to Cluster 0.")
else:
    print("This patient belongs to Cluster 1.")

# ============================================================
# CLUSTER CENTERS
# ============================================================

centers = pd.DataFrame(

    scaler.inverse_transform(kmeans.cluster_centers_),

    columns=X.columns

)

print("\nCluster Centers\n")

print(centers)