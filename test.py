import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import re

# Load the file
file_path = './Cleaned_TheWebster_2023_Final.csv'  # Update this path as needed
data = pd.read_csv(file_path)

# Make a copy of the data to preprocess
data_preprocessed = data.copy()

# Convert price columns to numeric by removing any non-numeric characters and converting to float
data_preprocessed['Item Retail Price'] = data_preprocessed['Item Retail Price'].apply(
    lambda x: float(re.sub(r'[^\d.]', '', x)) if isinstance(x, str) else x)
data_preprocessed['Purchased Amount'] = data_preprocessed['Purchased Amount'].apply(
    lambda x: float(re.sub(r'[^\d.]', '', x)) if isinstance(x, str) else x)

# Encode categorical columns using Label Encoding
label_encoders = {}
categorical_columns = ['Store', 'Transaction Type', 'Brand', 'Item Class', 'Item Subclass', 
                       'Item Department', 'Facet Type', 'Season', 'Sub Season']

for col in categorical_columns:
    le = LabelEncoder()
    data_preprocessed[col] = le.fit_transform(data_preprocessed[col])
    label_encoders[col] = le  # Save the encoder in case it's needed later for interpretation

# Select relevant features for clustering
features = ['Count', 'Entity ID', 'Store', 'Transaction Type', 'Brand', 
            'Item Class', 'Item Subclass', 'Item Department', 'Facet Type', 
            'Season', 'Sub Season', 'Item Retail Price', 'Purchased Amount', 'Purchased Units']
X = data_preprocessed[features]

# Apply K-Means clustering
n_clusters = 4  # You can adjust the number of clusters as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data_preprocessed['Cluster'] = kmeans.fit_predict(X)



# Evaluate the clustering
inertia = kmeans.inertia_  # Sum of squared distances of samples to their closest cluster center
silhouette_avg = silhouette_score(X, data_preprocessed['Cluster'])

print(f"Inertia (Within-cluster sum of squares): {inertia}")
print(f"Silhouette Score: {silhouette_avg}")

# Visualize clusters with PCA reduction to 2 dimensions
features = ['Count', 'Entity ID', 'Store', 'Transaction Type', 'Brand', 
            'Item Class', 'Item Subclass', 'Item Department', 'Facet Type', 
            'Season', 'Sub Season', 'Item Retail Price', 'Purchased Amount', 'Purchased Units']

X = data_preprocessed[features]

# Reduce dimensions to 2D for visualization with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data_preprocessed['Cluster'], cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label="Cluster")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clusters (PCA Reduced)")
plt.show()

data['Cluster'] =  data_preprocessed['Cluster']
cluster_summary = data.groupby('Cluster').mean(numeric_only=True)
print("Cluster Summary Statistics:")
print(cluster_summary)