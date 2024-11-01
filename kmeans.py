import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

DATA = "./Data/Cleaned_TheWebster_2022_Final.csv"

def load_data(file):
    dataset = pd.read_csv(file)
    return dataset

def eda(dataset):
    print(dataset.head())
    print("\nShape = ", dataset.shape)
    print("\nColumn names = \n", dataset.columns)
    print("\nInfo = \n", dataset.info())
    print("\nStats = \n", dataset.describe())
    print("\nCustomer count = ", dataset['Entity ID'].nunique())

def preprocess(dataset): 
    if (dataset.isnull().any().sum()) > 0:
        print("DATASET HAS NULL VALUES!")

    dataset["Entity ID"] = dataset["Entity ID"].astype(str)

    # Convert price columns to numeric
    dataset['Item Retail Price'] = dataset['Item Retail Price'].apply(
        lambda x: float(re.sub(r'[^\d.]', '', x)) if isinstance(x, str) else x
    )
    dataset['Purchased Amount'] = dataset['Purchased Amount'].apply(
        lambda x: float(re.sub(r'[^\d.]', '', x)) if isinstance(x, str) else x
    )

    dataset.drop(
        ['Count', 'Store', 'Transaction Type', 'Image URL'],
        axis = 1, 
        inplace = True
    )
    # How to split
    # men_data = dataset[dataset["Item Dept"] == 'Men']
    # women_data = dataset[dataset["Item Dept"] == 'Women']

    ###### Use if these columns are required #####
    # label_encoders = {}
    # categorical_columns = [
    #     'Brand', 'Item Class', 'Item Subclass', 
    #     'Item Department', 'Facet Type', 'Season', 
    #     'Sub Season'
    # ]

    # for col in categorical_columns:
    #     le = LabelEncoder()
    #     dataset[col] = le.fit_transform(dataset[col])
    #     label_encoders[col] = le  # Save the encoder in case its needed later for interpretation
    
    new_engg = feat_engg(data)
    new_engg_data = new_engg[0]
    attributes = new_engg[1]
    cleaned = rm_outlier(new_engg_data, attributes)
    scaled = scaling(cleaned, attributes)
    
    return scaled

def feat_engg(dataset):
    # Total Monetary
    amt_customer = dataset.groupby('Entity ID')['Purchased Amount'].sum()
    amt_customer = amt_customer.reset_index()

    # Frequency
    frequency = dataset['Entity ID'].value_counts().reset_index()
    frequency.columns = ['Entity ID', 'Frequency']

    amt_freq = pd.merge(amt_customer, frequency, on='Entity ID', how='inner')

    # Recency
    dataset['Transaction Date'] = pd.to_datetime(
        dataset['Transaction Date'],
        format='%Y-%m-%d'
    )
    max_date = max(dataset['Transaction Date'])
    dataset['Recency'] = max_date - dataset['Transaction Date']

    receny = dataset.groupby('Entity ID')['Recency'].min()
    receny = receny.reset_index()
    receny['Recency'] = receny['Recency'].dt.days

    # Merging all
    amt_freq_rec = pd.merge(amt_freq, receny, on='Entity ID', how='inner')
    attributes = ['Purchased Amount', 'Frequency', 'Recency']
    return amt_freq_rec, attributes
    

def rm_outlier(dataset, attributes):
    # plt.rcParams['figure.figsize'] = [10,8]
    # sns.boxplot(
    #     data = dataset[attributes],
    #     orient="v",
    #     palette="Set2",
    #     whis=1.5,
    #     saturation=1,
    #     width=0.7
    # )
    # plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')
    # plt.ylabel ("Range", fontweight = 'bold')
    # plt.xlabel ("Attributes", fontweight = 'bold')
    # plt.show()

    for att in attributes:
        Q1 = dataset[att].quantile(0.05)
        Q3 = dataset[att].quantile(0.95)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        dataset = dataset[(dataset[att] >= lower_bound) & (dataset[att] <= upper_bound)]

    return dataset

def scaling(dataset, attributes):
    dataset = dataset[attributes]
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(dataset)

    dataset_scaled = pd.DataFrame(dataset_scaled)

    dataset_scaled.columns = attributes

    return dataset

if __name__ == "__main__":
    data = load_data(DATA)
    # eda(data)

    preprocessed = preprocess(data)
    X = preprocessed.copy()

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=4, random_state=65)
    preprocessed['Cluster'] = kmeans.fit_predict(X)

    # Evaluate the clustering
    inertia = kmeans.inertia_  # Sum of squared distances of samples to their closest cluster center
    silhouette_avg = silhouette_score(X, preprocessed['Cluster'])

    print(f"Inertia (Within-cluster sum of squares): {inertia}")
    print(f"Silhouette Score: {silhouette_avg}")

    cluster_summary = preprocessed.groupby('Cluster').describe()
    print("Cluster Summary Statistics:")
    print(cluster_summary.T)

    # Reduce dimensions to 2D for visualization with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Plot the clusters
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=preprocessed['Cluster'], cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("K-Means Clusters (PCA Reduced)")
    plt.show()

    # For elbow method
    inertias = []
    for k in range(1, 11):  # Try cluster counts from 1 to 10
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.plot(range(1, 11), inertias, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show()
    