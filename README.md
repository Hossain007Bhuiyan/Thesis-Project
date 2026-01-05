# LNG Ferry Path Classification in the Baltic Sea

This repository contains the implementation and experiments from the master’s [thesis](https://su.diva-portal.org/smash/record.jsf?aq2=%5B%5B%5D%5D&c=67&af=%5B%5D&searchType=LIST_LATEST&sortOrder2=title_sort_asc&language=en&pid=diva2%3A1955718&aq=%5B%5B%5D%5D&sf=all&aqe=%5B%5D&sortOrder=author_sort_asc&onlyFullText=false&noOfRows=50&dswid=-2760)
**“Path Classification of Liquefied Natural Gas-Powered Ferries in the Baltic Sea”**.

## Overview
This project focuses on classifying ferry routes in the Baltic Sea using historical  
AIS (Automatic Identification System) data. Five LNG-powered ferries were analyzed  
over a six-month period to identify common navigation paths and route variations.

The main objective is to evaluate how different **time-series clustering algorithms**  
can group ferry trajectories into meaningful route clusters.

## Methods Used
Two unsupervised machine learning approaches were implemented and compared:

- **TimeSeriesKMeans with Dynamic Time Warping (DTW)**
- **Hierarchical Clustering with Fréchet Distance**

Each ferry trip is treated as a time-series of latitude and longitude coordinates.

## Key Findings
- Hierarchical clustering with **Fréchet Distance** produced clearer and more
  meaningful route clusters.
- TimeSeriesKMeans with **DTW** was faster but showed more overlapping clusters.
- AIS data requires strong preprocessing due to noise and trajectory anomalies.
- Different route clusters reveal operational differences such as speed and trip duration.

## Results & Visualizations
The following figures show examples of clustered ferry trajectories obtained from
different algorithms and similarity measures.

### Example: TimeSeriesKMeans (DTW)
![TimeSeriesKMeans Clustering](BalticSea-LNG-Ferry-Paths/clusters/mystar/kmeans/tal_hel_cluster_all_together.png)

### Example: Hierarchical Clustering (Fréchet Distance)
![Hierarchical Clustering](BalticSea-LNG-Ferry-Paths/clusters/mystar/hierarchical_clusters/tal_hel_all_together.png)

### Route Comparison
![Route Comparison](results/comparisons/route_comparison.png)

> 📌 **Note:**  
> All result images are generated from AIS trajectory data and stored inside the
> `results/` directory, organized by vessel and clustering method.

## Data
- Source: AIS data collected from VesselFinder
- Vessels: 5 LNG-powered ferries
- Time span: 6 months in the Baltic Sea

## Technologies
- Python
- Pandas, NumPy
- tslearn
- SciPy
- Matplotlib

## Purpose
This project demonstrates how unsupervised learning can be used to classify
maritime routes, supporting operational analysis and sustainable maritime planning.

## Thesis
This repository is based on a master’s thesis completed in **Spring 2024** at the  
**Department of Computer and Systems Sciences**.
