import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from fastdtw import fastdtw
from frechetdist import frdist
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from sklearn.metrics import silhouette_samples, silhouette_score
import sys

sys.setrecursionlimit(2000)

def load_and_preprocess_data(filepath, features):
    # Load data
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
        ts = pd.DataFrame(padded).interpolate(method='linear', limit_direction='both', axis=0).fillna(0).values
        padded_sequences.append(ts)
    
    return data, grouped, padded_sequences

# Function to compute DTW distance matrix
def compute_dtw_dist_matrix(series_list):
    n = len(series_list)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist, _ = fastdtw(series_list[i], series_list[j], dist=euclidean)
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    return dist_matrix

# Function to compute Fréchet distance matrix
def compute_frechet_dist_matrix(series_list):
    n = len(series_list)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = frdist(series_list[i], series_list[j])
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    return dist_matrix

# Compute distance matrix
def compute_distance_matrix(series_list, metric='dtw'):
    if metric == 'dtw':
        return compute_dtw_dist_matrix(series_list)
    elif metric == 'frechet':
        return compute_frechet_dist_matrix(series_list)

def compute_silhouette_scores(data, labels, clusters):
    silhouette_vals = silhouette_samples(data.reshape((data.shape[0], -1)), labels)
    cluster_silhouette_scores = []
    for i in range(clusters):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) > 1:
            cluster_silhouette_scores.append(np.mean(silhouette_vals[cluster_indices]))
        else:
            cluster_silhouette_scores.append(-1)
    return np.mean(silhouette_vals), cluster_silhouette_scores

def plot_clusters(data, grouped, clusters, cluster_count):
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'brown', 'pink', 'gray', 'cyan']
    
    # Plot clusters on a map
    fig, ax = plt.subplots(figsize=(12, 8))
    mystar = Basemap(projection='merc', llcrnrlon=24.6, llcrnrlat=59.3, urcrnrlon=25.2, urcrnrlat=60.2, lat_ts=20, resolution='i')
    peterpan_tra_tre = Basemap(projection='merc', llcrnrlon=10.812079683537172, llcrnrlat=53.936188126116654, urcrnrlon=13.331503817612624, urcrnrlat=55.41591913524446, lat_ts=20, resolution='i')
    peterpan_tre_swi = Basemap(projection='merc', llcrnrlon=13.020526107467903, llcrnrlat=55.396268879455995, urcrnrlon=14.666025335910248, urcrnrlat=53.84445028990433, lat_ts=20, resolution='i')
    nils_tre_ros = Basemap(projection='merc', llcrnrlon=11.866736685975777, llcrnrlat=54.06497438899161, urcrnrlon=13.331503817612624, urcrnrlat=55.41591913524446, lat_ts=20, resolution='i')
    nils_tre_swi = Basemap(projection='merc', llcrnrlon=12.603994448102485, llcrnrlat=53.799709865661654, urcrnrlon=14.730579796173291, urcrnrlat=55.52615670139893, lat_ts=20, resolution='i')
    nils_tra_ros = Basemap(projection='merc', llcrnrlon=10.71908045116268, llcrnrlat=53.93233860721343, urcrnrlon=12.243046501353131, urcrnrlat=54.37116558009551, lat_ts=20, resolution='i')
    vgrace = Basemap(projection='merc', llcrnrlon= 19.714084867523315, llcrnrlat = 59.81158988996984,  urcrnrlon= 22.273286743592408, urcrnrlat=60.540270444775324, lat_ts=20, resolution='i')
    vgrace_tur_sto = Basemap(projection='merc', llcrnrlon= 17.862903709730332, llcrnrlat = 59.15021451355056,  urcrnrlon= 22.273286743592408, urcrnrlat=60.540270444775324, lat_ts=20, resolution='i')
    vgrace_tur_sto.drawcoastlines()
    vgrace_tur_sto.fillcontinents(color='lightgray', lake_color='white')
    vgrace_tur_sto.drawmapboundary(fill_color='white')

    # Plot all clusters
    for cluster_num in range(1, cluster_count + 1):
        cluster_data = data[data['cluster'] == cluster_num]
        for trip_id in cluster_data['tripID'].unique():
            trip_data = cluster_data[cluster_data['tripID'] == trip_id]
            lon, lat = trip_data['LONGITUDE'].values, trip_data['LATITUDE'].values
            x, y = vgrace_tur_sto(lon, lat)
            vgrace_tur_sto.scatter(x, y, s=1, marker='o', color=colors[cluster_num - 1], label=f'Cluster {cluster_num}' if trip_id == cluster_data['tripID'].unique()[0] else "")
    
    plt.legend(loc='lower left')
    plt.title('AIS Time Series Clustering')
    plt.show()

    # Plot individual clusters
    for cluster_num in range(1, cluster_count + 1):
        fig, ax = plt.subplots(figsize=(12, 8))
        mystar = Basemap(projection='merc', llcrnrlon=24.6, llcrnrlat=59.3, urcrnrlon=25.2, urcrnrlat=60.2, lat_ts=20, resolution='i')
        peterpan_tra_tre = Basemap(projection='merc', llcrnrlon=10.812079683537172, llcrnrlat=53.936188126116654, urcrnrlon=13.331503817612624, urcrnrlat=55.41591913524446, lat_ts=20, resolution='i')
        peterpan_tre_swi = Basemap(projection='merc', llcrnrlon=13.020526107467903, llcrnrlat=55.396268879455995, urcrnrlon=14.666025335910248, urcrnrlat=53.84445028990433, lat_ts=20, resolution='i')
        nils_tre_ros = Basemap(projection='merc', llcrnrlon=11.866736685975777, llcrnrlat=54.06497438899161, urcrnrlon=13.331503817612624, urcrnrlat=55.41591913524446, lat_ts=20, resolution='i')
        nils_tre_swi = Basemap(projection='merc', llcrnrlon=12.603994448102485, llcrnrlat=53.799709865661654, urcrnrlon=14.730579796173291, urcrnrlat=55.52615670139893, lat_ts=20, resolution='i')
        nils_tra_ros = Basemap(projection='merc', llcrnrlon=10.71908045116268, llcrnrlat=53.93233860721343, urcrnrlon=12.243046501353131, urcrnrlat=54.37116558009551, lat_ts=20, resolution='i')
        vgrace = Basemap(projection='merc', llcrnrlon= 19.714084867523315, llcrnrlat = 59.81158988996984,  urcrnrlon= 22.273286743592408, urcrnrlat=60.540270444775324, lat_ts=20, resolution='i')
        vgrace_tur_sto = Basemap(projection='merc', llcrnrlon= 17.862903709730332, llcrnrlat = 59.15021451355056,  urcrnrlon= 22.273286743592408, urcrnrlat=60.540270444775324, lat_ts=20, resolution='i')
        vgrace_tur_sto.drawcoastlines()
        vgrace_tur_sto.fillcontinents(color='lightgray', lake_color='white')
        vgrace_tur_sto.drawmapboundary(fill_color='white')
        
        cluster_data = data[data['cluster'] == cluster_num]
        for trip_id in cluster_data['tripID'].unique():
            trip_data = cluster_data[cluster_data['tripID'] == trip_id]
            lon, lat = trip_data['LONGITUDE'].values, trip_data['LATITUDE'].values
            x, y = vgrace_tur_sto(lon, lat)
            vgrace_tur_sto.scatter(x, y, s=1, marker='o', color=colors[cluster_num - 1])
            
        plt.title(f'Cluster {cluster_num}')
        plt.show()

def hierarchical_clustering(filepath, cluster_count, metric, features):
    data, grouped, padded_sequences = load_and_preprocess_data(filepath, features)
    
    # Compute the chosen distance matrix
    dist_matrix = compute_distance_matrix(padded_sequences, metric=metric)
    
    # Perform hierarchical clustering using the chosen distance matrix
    condensed_dist_matrix = squareform(dist_matrix)
    linkage_matrix = linkage(condensed_dist_matrix, method='ward')
    
    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix, labels=np.arange(len(grouped)))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Index')
    plt.ylabel('Distance')
    plt.show()
    
    # Cut the dendrogram to form flat clusters
    clusters = fcluster(linkage_matrix, cluster_count, criterion='maxclust')
    
    # Assign clusters to the original data
    cluster_assignment = np.repeat(clusters, [len(sequence) for sequence in grouped])
    data['cluster'] = cluster_assignment

    # Ensure data_flat and labels_flat match in size
    padded_sequences_flat = np.vstack(padded_sequences)
    labels_flat = np.concatenate([np.repeat(cluster - 1, len(sequence)) for cluster, sequence in zip(clusters, padded_sequences)])
    
    # Compute silhouette scores using flattened data and labels
    avg_silhouette_score, cluster_silhouette_scores = compute_silhouette_scores(padded_sequences_flat, labels_flat, cluster_count)
    
    print(f"Average Silhouette Score: {avg_silhouette_score}")
    for i, score in enumerate(cluster_silhouette_scores):
        print(f"Cluster {i+1} Silhouette Score: {score}")
    
    # Compute additional statistics
    for i in range(1, cluster_count + 1):
        cluster_indices = np.where(cluster_assignment == i)[0]
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
        
        print(f"Cluster {i} - Number of Trips: {num_trips}, Average Speed: {avg_speed:.2f} knots, Average Travel Time: {avg_travel_time:.2f} hours")
    
    plot_clusters(data, grouped, clusters, cluster_count)

hierarchical_clustering('datasets/vikinggrace/turku_stockholm_cut.csv', cluster_count=2, metric='frechet', features=['LATITUDE', 'LONGITUDE', 'SPEED', 'HEADING'])
