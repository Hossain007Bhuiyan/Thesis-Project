# LNG Ferry Path Classification in the Baltic Sea

This repository contains the implementation and experiments from the master’s thesis:
**“Path Classification of Liquefied Natural Gas-Powered Ferries in the Baltic Sea”**  
Thesis link (DiVA): https://su.diva-portal.org/smash/record.jsf?aq2=%5B%5B%5D%5D&c=67&af=%5B%5D&searchType=LIST_LATEST&sortOrder2=title_sort_asc&language=en&pid=diva2%3A1955718&aq=%5B%5B%5D%5D&sf=all&aqe=%5B%5D&sortOrder=author_sort_asc&onlyFullText=false&noOfRows=50&dswid=-2760

---

## Overview
This project focuses on classifying ferry routes in the Baltic Sea using historical AIS
(Automatic Identification System) data. **Five LNG-powered ferries** were analyzed
over a **six-month** period to identify common navigation paths and route variations.

The main objective is to evaluate how different **time-series clustering algorithms**
can group ferry trajectories into meaningful route clusters.
---

## Methods Used
Two unsupervised machine learning approaches were implemented and compared:

- **TimeSeriesKMeans with Dynamic Time Warping (DTW)**
- **Hierarchical Clustering with Fréchet Distance**

Each one-way ferry trip is treated as a time-series of latitude and longitude coordinates.

---

## Key Findings
- Hierarchical clustering with **Fréchet Distance** produced clearer and more meaningful route clusters.
- TimeSeriesKMeans with **DTW** was faster but showed more overlapping clusters.
- AIS data requires strong preprocessing due to noise and trajectory anomalies.
- Different route clusters reveal operational differences such as speed and trip duration.

---

## Results & Visualizations
The following figures show examples of clustered ferry trajectories obtained from
different algorithms and similarity measures.

### Example: TimeSeriesKMeans (DTW)
![TimeSeriesKMeans Clustering](BalticSea-LNG-Ferry-Paths/clusters/mystar/kmeans/tal_hel_cluster_all_together.png)

### Example: Hierarchical Clustering (Fréchet Distance)
![Hierarchical Clustering](BalticSea-LNG-Ferry-Paths/clusters/mystar/hierarchical_clusters/tal_hel_all_together.png)

### Silhouette scores and optimal number of clusters

To assess clustering quality, we report the **silhouette score** for each route direction.  
Scores are shown for two feature groups: **Coordinate Similarity** and **Path Characteristics**.  
The number in parentheses indicates the **optimal number of clusters (k)** used for that result.

**Silhouette Scores and Optimal Clusters for the MyStar**
| From | To | Coordinate Similarity (k) | Path Characteristics (k) |
|---|---|---:|---:|
| Helsinki | Tallinn | 0.34 (2) | 0.11 (2) |
| Tallinn | Helsinki | 0.22 (2) | 0.29 (2) |


**Silhouette Scores and Optimal Clusters for the Peter Pan**
| From | To | Coordinate Similarity (k) | Path Characteristics (k) |
|---|---|---:|---:|
| Rostock | Travemunde | 0.12 (2) | 0.16 (3) |
| Rostock | Trelleborg | 0.025 (2) | 0.34 (2) |
| Swinoujscie | Trelleborg | 0.15 (2) | 0.02 (2) |
| Travemunde | Rostock | 0.26 (2) | 0.23 (2) |
| Travemunde | Trelleborg | 0.96 (2) | 0.74 (2) |
| Trelleborg | Rostock | 0.0 (2) | 0.08 (2) |
| Trelleborg | Swinoujscie | 0.92 (2) | 0.88 (2) |
| Trelleborg | Travemunde | 0.95 (2) | 0.52 (2) |

**Silhouette Scores and Optimal Clusters for the Nils Holgersson**
| From | To | Coordinate Similarity (k) | Path Characteristics (k) |
|---|---|---:|---:|
| Rostock | Travemunde | 0.07 (2) | 0.14 (3) |
| Rostock | Trelleborg | 0.0 (2) | 0.19 (2) |
| Swinoujscie | Trelleborg | 0.03 (2) | 0.16 (2) |
| Travemunde | Rostock | 0.20 (2) | 0.34 (2) |
| Travemunde | Trelleborg | 0.08 (2) | 0.55 (5) |
| Trelleborg | Rostock | 0.06 (2) | 0.6 (2) |
| Trelleborg | Swinoujscie | 0.88 (2) | 0.87 (2) |
| Trelleborg | Travemunde | 0.16 (2) | 0.05 (2) |

**Silhouette Scores and Optimal Clusters for the Viking Grace**
| From | To | Coordinate Similarity (k) | Path Characteristics (k) |
|---|---|---:|---:|
| Mariehamn | Turku | 0.0 (2) | 0.22 (2) |
| Stockholm | Mariehamn | 0.01 (2) | 0.03 (2) |
| Turku | Stockholm | -0.01 (2) | 0.94 (2) |

### Silhouette Scores and Optimal Clusters for the Viking Glory
| From | To | Coordinate Similarity (k) | Path Characteristics (k) |
|---|---|---:|---:|
| Mariehamn | Stockholm | 0.03 (2) | 0.02 (2) |
| Stockholm | Turku | 0.0 (2) | 0.08 (2) |
| Turku | Mariehamn | 0.0 (2) | 0.29 (2) |


### Route Comparison
![Route Comparison](results/comparisons/route_comparison.png)

> 📌 **Note:**  
> Result images are generated from AIS trajectory data and stored inside the `clusters/` directory,
> organized by vessel and clustering method.

---

## Data
- **Type:** AIS data (historical vessel tracking messages)
- **Region:** Baltic Sea
- **Time span:** 6 months
- **Vessels:** 5 LNG-powered ferries

### Data Source
- **Provider:** VesselFinder
- **Website:** https://www.vesselfinder.com/

> ⚠️ **Data availability / license note:**  
> The raw AIS data used in this thesis was collected from VesselFinder and may be subject to their
> data usage terms. Therefore, the raw dataset is **not included** in this repository.  
> This repo contains the **code** and **generated result images**, and the pipeline can be reproduced
> by collecting the same vessels’ AIS data from VesselFinder (or another AIS provider).

---

## Project Setup (Prepare)

### Requirements
- Python 3.9+ (recommended)
- Libraries:
  - tslearn
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - scipy

### Create a virtual environment (recommended)

**Windows (PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate
