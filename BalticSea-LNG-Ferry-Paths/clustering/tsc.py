import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset
from sklearn.metrics import silhouette_samples, silhouette_score
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
    
    time_series_data = to_time_series_dataset(padded_sequences)
    
    # Fill NaN values with interpolation or zero
    for i in range(time_series_data.shape[0]):
        for j in range(time_series_data.shape[2]):
            ts = pd.Series(time_series_data[i, :, j])
            time_series_data[i, :, j] = ts.interpolate(limit_direction='both', method='linear').fillna(0)
    
    # Normalize the time series data
    scaler = TimeSeriesScalerMeanVariance()  # Standardizes to have mean 0 and variance 1
    time_series_data_scaled = scaler.fit_transform(time_series_data)
    
    return data, grouped, time_series_data_scaled

def cluster_with_dtw(time_series_data_scaled, clusters, max_iter):
    # Clustering configuration using DTW
    model = TimeSeriesKMeans(n_clusters=clusters, metric="dtw", max_iter=max_iter)
    
    # Fit the model
    labels = model.fit_predict(time_series_data_scaled)
    return labels

def plot_clusters(data, grouped, labels, clusters):
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'brown', 'pink', 'gray', 'cyan']

    # Plot clusters on a map
    fig, ax = plt.subplots(figsize=(12, 8)) 
    mystar = Basemap(projection='merc', llcrnrlon=24.6, llcrnrlat=59.3, urcrnrlon=25.2, urcrnrlat=60.2, lat_ts=20, resolution='i')
    peterpan = Basemap(projection='merc', llcrnrlon=12.819568000917554, llcrnrlat = 53.857674615525,  urcrnrlon=14.809829511730834, urcrnrlat=55.49736106903503, lat_ts=20, resolution='i')
    vgrace = Basemap(projection='merc', llcrnrlon= 19.714084867523315, llcrnrlat = 59.81158988996984,  urcrnrlon= 22.273286743592408, urcrnrlat=60.540270444775324, lat_ts=20, resolution='i')
    vgrace.drawcoastlines()
    vgrace.fillcontinents(color='lightgray', lake_color='white')
    vgrace.drawmapboundary(fill_color='white')

    # Plot all clusters
    for i in range(clusters):
        cluster_indices = np.where(labels == i)[0]
        for idx in cluster_indices:
            coordinates = np.array(grouped[idx])
            lon, lat = coordinates[:, 1], coordinates[:, 0]
            x, y = vgrace(lon, lat)
            vgrace.scatter(x, y, s=1, marker='o', color=colors[i % len(colors)], label=f'Cluster {i+1}' if idx == cluster_indices[0] else "")

    plt.legend(loc='lower left')
    plt.title('All Clusters Together')
    plt.show()

    # Plot individual clusters
    for i in range(clusters):
        fig, ax = plt.subplots(figsize=(12, 8))
        mystar = Basemap(projection='merc', llcrnrlon=24.6, llcrnrlat=59.3, urcrnrlon=25.2, urcrnrlat=60.2, lat_ts=20, resolution='i')
        peterpan = Basemap(projection='merc', llcrnrlon=12.819568000917554, llcrnrlat = 53.857674615525,  urcrnrlon=14.809829511730834, urcrnrlat=55.49736106903503, lat_ts=20, resolution='i')
        vgrace = Basemap(projection='merc', llcrnrlon= 19.714084867523315, llcrnrlat = 59.81158988996984,  urcrnrlon= 22.273286743592408, urcrnrlat=60.540270444775324, lat_ts=20, resolution='i')
        vgrace.drawcoastlines()
        vgrace.fillcontinents(color='lightgray', lake_color='white')
        vgrace.drawmapboundary(fill_color='white')
        
        cluster_indices = np.where(labels == i)[0]
        for idx in cluster_indices:
            coordinates = np.array(grouped[idx])
            lon, lat = coordinates[:, 1], coordinates[:, 0]
            x, y = vgrace(lon, lat)
            vgrace.scatter(x, y, s=1, marker='o', color=colors[i % len(colors)])
            
        plt.title(f'Cluster {i+1}')
        plt.show()

def compute_silhouette_scores(time_series_data, labels, clusters):
    # Compute silhouette score for each cluster
    silhouette_vals = silhouette_samples(time_series_data.reshape((time_series_data.shape[0], -1)), labels)
    cluster_silhouette_scores = []
    for i in range(clusters):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) > 1:  # Only compute silhouette score if there are more than 1 sample in the cluster
            cluster_silhouette_scores.append(np.mean(silhouette_vals[cluster_indices]))
        else:
            cluster_silhouette_scores.append(-1)  # Not enough data to compute silhouette score

    return np.mean(silhouette_vals), cluster_silhouette_scores

def iterative_clustering(filepath, features, clusters):
    data, grouped, time_series_data_scaled = load_and_preprocess_data(filepath, features)
    
    max_iter = 100
    max_iter_limit = 600
    best_silhouette_score = -1
    best_labels = None
    best_max_iter = max_iter
    best_cluster_sil_scores = []

    while max_iter <= max_iter_limit:
        labels = cluster_with_dtw(time_series_data_scaled, clusters, max_iter)
        avg_silhouette_score, cluster_sil_scores = compute_silhouette_scores(time_series_data_scaled, labels, clusters)
        
        print(f"Max Iter: {max_iter}, Avg Silhouette Score: {avg_silhouette_score}")
        for i, score in enumerate(cluster_sil_scores):
            print(f"Cluster {i+1} Silhouette Score: {score}")
        
        if avg_silhouette_score > best_silhouette_score:
            best_silhouette_score = avg_silhouette_score
            best_labels = labels
            best_max_iter = max_iter
            best_cluster_sil_scores = cluster_sil_scores
            max_iter += 100
        else:
            break

    print(f"Best Max Iter: {best_max_iter}, Best Avg Silhouette Score: {best_silhouette_score}")
    for i, score in enumerate(best_cluster_sil_scores):
        print(f"Cluster {i+1} Silhouette Score: {score}")

    # Compute and print additional statistics
    for i in range(clusters):
        cluster_indices = np.where(best_labels == i)[0]
        num_trips = len(cluster_indices)
        
        # Filter data for the current cluster trips
        cluster_trip_ids = data.loc[data.index.isin(cluster_indices), 'tripID'].unique()
        cluster_data = data[data['tripID'].isin(cluster_trip_ids)]
        
        avg_speed = cluster_data['SPEED'].mean()
        
        travel_times = []
        for tripID in cluster_trip_ids:
            trip_data = cluster_data[cluster_data['tripID'] == tripID]
            if len(trip_data) > 0:
                start_time = trip_data['DATE TIME (UTC)'].min()
                end_time = trip_data['DATE TIME (UTC)'].max()
                travel_times.append((end_time - start_time).total_seconds() / 3600.0)
        avg_travel_time = np.mean(travel_times)

        print(f"Cluster {i+1} - Number of Trips: {num_trips}, Average Speed: {avg_speed:.2f} knots, Average Travel Time: {avg_travel_time:.2f} hours")

    plot_clusters(data, grouped, best_labels, clusters)


#iterative_clustering('datasets/mystar/hel_tal.csv', features=['LATITUDE', 'LONGITUDE'], clusters=2)
# iterative_clustering('datasets/mystar/tal_hel.csv', features=['LATITUDE', 'LONGITUDE'], clusters=2)
# iterative_clustering('datasets/mystar/tal_hel.csv', features=['LATITUDE', 'LONGITUDE', 'SPEED', 'HEADING'], clusters=2)
# iterative_clustering('datasets/peterpan/rostock_trelleborg.csv', features=['LATITUDE', 'LONGITUDE', 'SPEED', 'HEADING'], clusters=2)
iterative_clustering('datasets/vikingglory/turku_mariehamn.csv', features=['LATITUDE', 'LONGITUDE', 'SPEED', 'HEADING'], clusters=2)
