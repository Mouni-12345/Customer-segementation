import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Create a sample dataset
data = {
    'Age': [22, 25, 47, 52, 46, 56, 55, 60, 62, 61, 18, 28, 27, 32],
    'Income': [50000, 55000, 65000, 70000, 75000, 80000, 60000, 70000, 75000, 55000, 40000, 45000, 50000, 60000],
    'Spending': [2000, 2500, 3000, 3500, 4000, 4500, 3000, 3500, 4000, 2500, 1500, 2000, 2500, 3000]
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Scale the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_scaled)
labels = kmeans.labels_

# Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(df_scaled[:, 0], df_scaled[:, 1], c=labels)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.title('Customer Segmentation')
plt.xlabel('Age')
plt.ylabel('Income/Spending')
plt.legend()
plt.show()

# Analyze the clusters
df['Cluster'] = labels
print(df.groupby('Cluster').mean())