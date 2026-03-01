# The State of Brazilian Favelas

**Infrastructural Regimes and Urban Inequality**

This repository contains the analysis code and data processing pipelines for the research paper *"The State of Brazilian Favelas: Infrastructural Regimes and Urban Inequality"* by Silva et al. The study provides a national-scale assessment of infrastructural heterogeneity across more than 12,000 informal settlements (favelas) in Brazil, using the complete set of favela boundaries from the 2022 Brazilian Census.

## Overview

Brazilian favelas are a central component of the country's urban fabric, yet their infrastructural conditions remain poorly characterized at national scale. Although frequently treated as a homogeneous category in statistics and policy frameworks, favelas exhibit substantial internal variation in access to basic services, housing conditions, and environmental exposure.

This study:

- Integrates census-based housing and demographic indicators with environmental, topographic, pollution, and accessibility data
- Constructs four UN-Habitat–aligned indices: **improved water access**, **improved sanitation access**, **sufficient living area**, and **housing durability**
- Identifies three distinct **infrastructural regimes**—Structured, Partially Structured, and Unstructured—distributed unevenly across Brazil
- Examines socio-environmental correlates and predictability of infrastructure access
- Tests whether urban scaling laws apply at the scale of informal settlements (they do not)

## Key Findings

- **Three regimes**: 49.7% of favelas are *Structured* (high water and sanitation), 37.1% *Partially Structured* (high water, lower sanitation), and 13.2% *Unstructured* (low access to both)
- **Regional concentration**: Unstructured favelas are strongly concentrated in the North; the Southeast and South show predominantly Structured settlements
- **Water and sanitation** drive regime differentiation; housing conditions and population size play a secondary role
- **No scaling relationships**: Population size shows no meaningful association with infrastructural outcomes—contrary to urban scaling theory at city scale
- **Predictability**: Non-infrastructural factors (deprivation, air pollution, topography, accessibility) explain ~91–92% of variance in water and sanitation indices

## Interactive Dashboard

Explore the favela data and infrastructural regimes through an interactive map:

**[favela-map.vercel.app/map](https://favela-map.vercel.app/map)**

The dashboard allows you to visualize favela boundaries across Brazil, toggle between satellite and standard basemaps, and explore the spatial distribution of infrastructural conditions by layer.

## Project Structure

```
src/
├── pre-processing/          # Data preparation and harmonization
│   ├── censitary_sectors.ipynb
│   ├── censitary_sectors_interpolation.ipynb
│   ├── sentinel_5P.ipynb    # Air pollution (NO2, SO2, CO, O3)
│   ├── ibp.ipynb            # Brazilian Deprivation Index (CIDACS)
│   └── hospitals_distances.ipynb
├── cluster/
│   └── generate_clusters.ipynb   # K-means clustering (k=3)
└── analysis/
    ├── simple_corr.ipynb    # Correlation analyses
    ├── multi_corr.ipynb
    ├── boxplot.ipynb        # Regime comparisons
    ├── gini.ipynb            # Intra-regime inequality
    ├── predict.ipynb         # Random Forest, SHAP, regime prediction
    └── urban_scaling_laws.ipynb  # Scaling analysis (population vs indices)
```

## Data Sources

| Scale | Domain | Sources |
|-------|--------|---------|
| **Household** | Census, socio-economic | 2022 Brazilian Census (IBGE), Brazilian Deprivation Index (CIDACS) |
| **Within-area** | Environmental | MapBiomas, DEM (slope, HAND), Sentinel-5P, Landsat (GEE) |
| **Area-connect** | Accessibility | AGSN 2019 (distances to hospitals, primary care) |

All data are harmonized to the favela polygons (SIRGAS 2000) delineated by the 2022 Census.

## Requirements

- Python 3.x
- Key dependencies: `geopandas`, `pandas`, `numpy`, `scikit-learn`, `shap`, `xgboost`, `statsmodels`, `seaborn`, `matplotlib`, `kneed`

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt  # if available
```

## Citation

If you use this code or findings, please cite:

> Silva, J.P., Porto de Albuquerque, J., Pedrassoli, J., Rodrigues-Jr, J.F. *The State of Brazilian Favelas: Infrastructural Regimes and Urban Inequality.*

## Authors

- **João P. Silva** (jp.silva@usp.br) — ICMC, University of São Paulo
- João Porto de Albuquerque — University of Glasgow
- Julio Pedrassoli — ICMC, University of São Paulo
- Jose F. Rodrigues-Jr — ICMC, University of São Paulo

## License

[Specify license if applicable]
