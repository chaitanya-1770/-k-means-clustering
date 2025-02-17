import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load the dataset
@st.cache
def load_data():
    # Load the environmental factors dataset
    data = pd.read_csv('environmental factors.csv')
    return data

# Function to perform KMeans clustering
def apply_kmeans(data, n_clusters):
    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(data_scaled)

    # Calculate silhouette score
    sil_score = silhouette_score(data_scaled, data['cluster'])
    return data, sil_score

# Function to create a scatter plot
def create_plot(data):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='carbon_emissions', y='pollution_level', hue='cluster', data=data, palette='viridis', s=100, alpha=0.7, edgecolor='k')
    plt.title('K-Means Clustering of Environmental Factors')
    plt.xlabel('Carbon Emissions')
    plt.ylabel('Pollution Level')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt)

# Streamlit UI for the app
def main():
    st.title('K-Means Clustering for Environmental Factors')

    # Load data
    data = load_data()

    # Display dataset
    st.write('Dataset preview:')
    st.write(data.head())

    # Number of clusters
    n_clusters = st.slider('Select the number of clusters (K)', min_value=2, max_value=10, value=2, step=1)

    # Apply KMeans clustering
    clustered_data, sil_score = apply_kmeans(data.copy(), n_clusters)

    # Display silhouette score
    st.write(f'Silhouette Score: {sil_score:.4f}')

    # Plot the clustering results
    create_plot(clustered_data)

if __name__ == '__main__':
    main()
