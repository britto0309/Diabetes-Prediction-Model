# Diabetes Patient Clustering using K-Means Clustering

## Overview

This project groups diabetes patients into different clusters using the **K-Means Clustering** algorithm. Unlike supervised learning, K-Means does not use the target variable during training. The model identifies patients with similar health characteristics based on medical attributes such as glucose level, blood pressure, BMI, insulin level, and age. The project also demonstrates data preprocessing, feature scaling, cluster analysis, and data visualization.

## Objectives

- Load and explore the diabetes dataset.
- Handle invalid values in the dataset.
- Standardize numerical features using StandardScaler.
- Determine the optimal number of clusters using the Elbow Method.
- Train a K-Means Clustering model.
- Group patients based on similar medical characteristics.
- Visualize patient clusters using PCA.
- Predict the cluster for a new patient based on user inputs.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn

## Dataset

The dataset contains medical information about diabetes patients, including:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (Used only for comparison after clustering)

## Methodology

The project follows a complete machine learning workflow:

- Loaded the diabetes dataset using Pandas.
- Replaced invalid zero values in selected medical features.
- Standardized the dataset using StandardScaler.
- Used the Elbow Method to determine the optimal number of clusters.
- Trained a K-Means Clustering model.
- Reduced the data to two dimensions using PCA for visualization.
- Assigned each patient to a cluster.
- Compared the generated clusters with the actual Outcome values.
- Predicted the cluster for new patient data entered by the user.

## Results

The model provides insights such as:

- Identification of patient groups with similar medical characteristics.
- Optimal number of clusters using the Elbow Method.
- Cluster distribution of diabetes patients.
- Cluster centroids representing average patient characteristics.
- Visualization of patient clusters using PCA.

## Future Improvements

- Experiment with different numbers of clusters.
- Compare K-Means with other clustering algorithms such as DBSCAN and Hierarchical Clustering.
- Use larger healthcare datasets for improved clustering analysis.
- Develop an interactive dashboard to visualize cluster information.

## Author

**Britto Domnic Aro J**

Aspiring Machine Learning Engineer | Python Developer | Data Analytics Enthusiast
