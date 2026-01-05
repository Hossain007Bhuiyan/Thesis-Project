import pandas as pd
import numpy as np
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def load_and_preprocess_data(filepath, features):

    data = pd.read_csv(filepath)
    
    # Convert 'DATE TIME (UTC)' to datetime format for better handling
    data['DATE TIME (UTC)'] = pd.to_datetime(data['DATE TIME (UTC)'])
    data.sort_values(['tripID', 'DATE TIME (UTC)'], inplace=True)
    
    # Group by tripID and create an array for each trip
    grouped = data.groupby('tripID')[features].apply(lambda df: df.values.tolist()).tolist()
    
    # Find the maximum length of the time series
    max_length = max(map(len, grouped))
    
    # Pad sequences to have the same length and create a uniform time series dataset
    padded_sequences = []
    for sequence in grouped:
        padded = np.array(sequence + [[np.nan] * len(features)] * (max_length - len(sequence)))
        padded_sequences.append(padded)
    
    # Convert to time series dataset
    time_series_data = to_time_series_dataset(padded_sequences)
    
    # Fill NaN values with interpolation or zero
    for i in range(time_series_data.shape[0]):
        for j in range(time_series_data.shape[2]):
            ts = pd.Series(time_series_data[i, :, j])
            time_series_data[i, :, j] = ts.interpolate(limit_direction='both', method='linear').fillna(0)
    
    return time_series_data

# Function to perform clustering and compute average silhouette scores over 3 iterations
def compute_avg_silhouette_scores(time_series_data, max_clusters, max_iter, n_iterations=3):
    avg_sil_scores = np.zeros(max_clusters - 1)  # To store average silhouette scores for clusters 2 to max_clusters

    for _ in range(n_iterations):
        sil_scores = []

        for k in range(2, max_clusters + 1):
            model = TimeSeriesKMeans(n_clusters=k, metric='dtw', max_iter=max_iter)
            labels = model.fit_predict(time_series_data)
            
            # Compute silhouette score
            sil_score = silhouette_score(time_series_data.reshape((time_series_data.shape[0], -1)), labels)
            sil_scores.append(sil_score)
        
        avg_sil_scores += np.array(sil_scores)

    avg_sil_scores /= n_iterations  # Calculate the average silhouette scores

    optimal_clusters = np.argmax(avg_sil_scores) + 2  # Adding 2 because range starts from 2
    return optimal_clusters, avg_sil_scores.tolist()

def plot_silhouette_scores(sil_scores):
    # Plot silhouette scores as bar graph
    plt.figure()
    plt.plot(range(2, len(sil_scores) + 2), sil_scores, marker='o')    
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average Silhouette Score')
    plt.title('Average Silhouette Scores for Different Cluster Counts')
    plt.show()

# Parameters
#filepath = 'datasets/mystar/tal_hel.csv'
filepath = 'datasets/vikingglory/turku_mariehamn.csv'
features = ['LATITUDE', 'LONGITUDE']
max_clusters = 5
max_iter = 200

# Load and preprocess data
time_series_data = load_and_preprocess_data(filepath, features)

# Compute average silhouette scores
optimal_clusters, avg_sil_scores = compute_avg_silhouette_scores(time_series_data, max_clusters=max_clusters, max_iter=max_iter)

# Plot average silhouette scores
plot_silhouette_scores(avg_sil_scores)
