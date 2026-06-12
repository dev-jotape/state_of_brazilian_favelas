# The State of Brazilian Favelas

**Infrastructural Regimes and Urban Inequality**

This repository contains the analysis code and data-processing pipelines for the
research paper *"The State of Brazilian Favelas: Infrastructural Regimes and Urban
Inequality."* The study provides a national-scale assessment of infrastructural
heterogeneity across more than 12,000 informal settlements (favelas) in Brazil,
using the complete set of favela boundaries from the 2022 Brazilian Census.

## Overview

Although frequently treated as a homogeneous category in statistics and policy,
favelas exhibit substantial internal variation in access to basic services,
housing conditions, and environmental exposure. This study:

- Integrates census-based housing and demographic indicators with environmental,
  topographic, pollution, and accessibility data
- Constructs four UN-Habitat–aligned indices: **improved water access**,
  **improved sanitation access**, **sufficient living area**, **housing durability**
- Identifies three **infrastructural regimes** — Structured, Partially Structured,
  Unstructured — distributed unevenly across Brazil
- Examines socio-environmental correlates and predictability of infrastructure access

## Key Findings

- **Three regimes:** 49.7% Structured, 37.1% Partially Structured, 13.2% Unstructured
- **Regional concentration:** Unstructured favelas concentrated in the North; the
  Southeast and South are predominantly Structured
- **Water and sanitation** drive regime differentiation; housing and population size
  play a secondary role
- **Predictability:** non-infrastructural factors explain ~91–92% of variance in the
  water and sanitation indices

## Interactive Dashboard

**[favela-map.vercel.app/map](https://favela-map.vercel.app/map)** — visualize favela
boundaries and infrastructural conditions by layer.

---

## 1. System requirements

### Software dependencies and versions
- **Operating systems tested:** Pop!_OS 24.04 LTS (Ubuntu 24.04 LTS based)
- **Python:** 3.13.12
- **Key dependencies** (pinned versions in `requirements.txt`):
  `earthengine-api`, `geopandas`, `pandas`, `numpy`, `scikit-learn`, `shap`,
  `xgboost`, `statsmodels`, `seaborn`, `matplotlib`, `kneed`

### Non-standard hardware
None required. All analyses run on a standard desktop CPU.

---

## 2. Installation guide

### Instructions
```bash
git clone https://github.com/dev-jotape/state_of_brazilian_favelas.git
cd state_of_brazilian_favelas
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For the Sentinel-5P extraction, authenticate once with Google Earth Engine:
```bash
earthengine authenticate
```

---

## 3. Project structure

```
src/
├── pre-processing/                    # Data preparation and harmonization
│   ├── censitary_sectors.ipynb
│   ├── censitary_sectors_interpolation.ipynb
│   ├── sentinel_5P.ipynb              # Air pollution (NO2, SO2, CO, O3) via Earth Engine
│   ├── ibp.ipynb                      # Brazilian Deprivation Index (CIDACS)
│   └── hospitals_distances.ipynb      # Euclidean distances to health facilities
├── cluster/
│   └── generate_clusters.ipynb        # K-means clustering (k=3)
└── analysis/
    ├── simple_corr.ipynb
    ├── multi_corr.ipynb
    ├── boxplot.ipynb                  # Regime comparisons
    ├── gini.ipynb                     # Intra-regime inequality
    ├── predict.ipynb                  # Random Forest, SHAP, regime prediction
    ├── urban_scaling_laws.ipynb
    └── train_bootstrap_features.py    # Bootstrapped classifiers on EfficientNetV2-S TTA features
```
> The morphological (image-based EfficientNetV2 + SVM) analysis is implemented in
> `src/analysis/train_bootstrap_features.py`.

---

## 4. Instructions for use

The analyses reported in the manuscript can be reproduced using the complete dataset available on Mendeley Data.

### Setup

1. Clone the repository and install the required dependencies (Section 2).
2. Download the integrated dataset from Mendeley Data and place it under the dataset/ directory.
3. Authenticate with Google Earth Engine if you intend to reproduce the environmental data extraction steps.

### Workflow

#### Data preprocessing (optional)

The repository includes notebooks to reproduce the harmonization and integration of the original data sources:

text src/pre-processing/ ├── censitary_sectors.ipynb ├── censitary_sectors_interpolation.ipynb ├── sentinel_5P.ipynb ├── ibp.ipynb └── hospitals_distances.ipynb 

These steps are optional because the fully integrated dataset is already available through Mendeley Data.

#### Infrastructure regime identification

Run:

text src/cluster/generate_clusters.ipynb 

This notebook reproduces:

- Indicator normalization
- Infrastructure index construction
- K-means clustering (k = 3)
- Cluster validation analyses
- Regime assignment

#### Statistical analyses

Run the notebooks in:

text src/analysis/ 

These notebooks reproduce:

- Correlation analyses
- Gini coefficient analyses
- Regime comparisons
- Predictive modelling
- SHAP explanations
- Urban scaling analyses

#### Morphological-signal analysis

Run:

text src/analysis/train_bootstrap_features.py 

This script reproduces the image-based classification experiments using EfficientNetV2 feature embeddings and Support Vector Machine (SVM) classifiers.

### Expected outputs

Executing the workflow reproduces the main results reported in the manuscript, including:

- Four normalized infrastructure indices:
  - Improved water access
  - Improved sanitation access
  - Sufficient living area
  - Housing durability

- Three infrastructural regimes:
  - Structured
  - Partially Structured
  - Unstructured

- Regional distribution analyses

- Predictive modelling and SHAP explanations

- Morphological-signal experiments

- Figures and tables reported throughout the manuscript.

---

## 5. Reproducibility

The repository includes fixed random seeds for clustering, train-test splits, and bootstrap experiments whenever applicable. Running the workflow on the dataset provided through Mendeley Data should reproduce the reported results within expected numerical precision.

---

## 6. Data availability

The integrated dataset is publicly available through Mendeley Data:

https://doi.org/10.17632/x959s4g493.1

The repository includes:

- Favela polygon shapefile
- Integrated socio-demographic, environmental, and accessibility indicators
- Data dictionary
- EfficientNetV2 image feature embeddings used in the morphological analysis

The original satellite imagery cannot be redistributed due to Google Maps licensing restrictions. Only the derived image embeddings used in the experiments are provided.

Primary data sources include:

| Scale | Domain | Sources |
|---------|---------|---------|
| Household | Census and socio-economic conditions | 2022 Census (IBGE), IBP (CIDACS) |
| Within-area | Environmental conditions | MapBiomas, Sentinel-5P (via Google Earth Engine) |
| Area-connect | Accessibility and infrastructure | AGSN 2019, health facility datasets |

All datasets were harmonized to the 2022 Census favela polygons using the SIRGAS 2000 coordinate reference system.

---

## 7. License

### Code

Released under the Apache License 2.0.

### Dataset

Released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

See the LICENSE file for details.

---

## 8. Citation

> Silva, J.P., Porto de Albuquerque, J., Pedrassoli, J.C., Spadon, G.,
> Rodrigues-Jr, J.F. *The State of Brazilian Favelas: Infrastructural Regimes and
> Urban Inequality* (2026).

**Dataset:**
> da Silva, João Pedro (2026), "The State of Brazilian Favelas", Mendeley Data, V1,
> doi: [10.17632/x959s4g493.1](https://doi.org/10.17632/x959s4g493.1)

## Authors

- **João Pedro da Silva** (jp.silva@usp.br) — ICMC, University of São Paulo
- João Porto de Albuquerque — School of Social and Political Sciences, University of Glasgow
- Julio Cesar Pedrassoli — Department of Geography, University of São Paulo
- Gabriel Spadon — Faculty of Computer Science, Dalhousie University
- Jose F. Rodrigues-Jr — ICMC, University of São Paulo
