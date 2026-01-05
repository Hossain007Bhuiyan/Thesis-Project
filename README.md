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
