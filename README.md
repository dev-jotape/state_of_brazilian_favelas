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
- Tests whether these regimes leave a measurable **morphological signal** in
  high-resolution satellite imagery

## Key Findings

- **Three regimes:** 49.7% Structured, 37.1% Partially Structured, 13.2% Unstructured
- **Regional concentration:** Unstructured favelas concentrated in the North; the
  Southeast and South are predominantly Structured
- **Water and sanitation** drive regime differentiation; housing and population size
  play a secondary role
- **Predictability:** non-infrastructural factors explain ~91–92% of variance in the
  water and sanitation indices (Random Forest with RFE), and predict regime
  membership with 0.72 overall accuracy
- **Morphological signal:** the Structured regime is clearly separable from
  satellite-image features alone (F1 = 0.883); the Unstructured and Partially
  Structured regimes are progressively harder to distinguish

## Interactive Dashboard

**[favela-map.vercel.app/map](https://favela-map.vercel.app/map)** — visualize favela
boundaries and infrastructural conditions by layer.

---

## 1. System requirements

### Software dependencies and versions
- **Operating systems tested:** Pop!_OS 24.04 LTS (Ubuntu 24.04 LTS based)
- **Python:** 3.13.12
- **Key dependencies** (exact versions pinned in `requirements.txt`):
  `earthengine-api==1.5.8`, `geopandas==1.0.1`, `shapely==2.0.7`, `tobler==0.12.1`,
  `numpy==2.2.4`, `pandas==2.2.3`, `scipy==1.15.2`, `scikit-learn==1.6.1`,
  `xgboost==3.0.2`, `shap==0.48.0`, `tensorflow==2.21.0`, `statsmodels==0.14.4`,
  `seaborn==0.13.2`, `matplotlib==3.10.1`, `plotly>=5.24.1,<6.0.0`, `kneed==0.8.5`

### Non-standard hardware
None required. All analyses run on a standard desktop CPU. The EfficientNetV2
feature extraction in `src/analysis/train_bootstrap_features.py` (morphological-signal
analysis, Section 2.5 of the manuscript) benefits from a CUDA-capable GPU but does
not require one.

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

**Typical install time:** a few minutes on a standard desktop computer with a normal
internet connection.

For the Sentinel-5P air-pollution extraction, authenticate once with Google Earth
Engine (only needed to re-run the environmental pre-processing step):
```bash
earthengine authenticate
```

---

## 3. Demo

The analysis is a multi-stage geospatial and machine-learning pipeline rather than a
single command. Because the input layers are large (the integrated favela shapefile
alone is ~28 MB, covering 12,344 settlements), the example data required to run each
stage is distributed through **Mendeley Data**
([DOI 10.17632/x959s4g493.2](https://doi.org/10.17632/x959s4g493.2)) rather than bundled
in this repository. Download it and place it under `dataset/` following the layout in
Section 4, then run the stages below.

Each stage lists its input, expected output, and expected run time on a standard
desktop CPU:

| Stage | Command | Input | Expected output | Expected run time |
|-------|---------|-------|-----------------|-------------------|
| Index construction & clustering | `src/cluster/generate_clusters.ipynb` | Integrated per-favela indicator table (`dataset/favelas_urban_communities/slum_variables/final/`) | Four normalized indices + `k=3` regime labels. Regime split ≈ **49.7% Structured / 37.1% Partially Structured / 13.2% Unstructured**; silhouette peaks at k=3 (≈0.530) | ~1–3 min |
| Join & finalize dataset | `src/cluster/join_cities_data.ipynb` | Clustered favelas + municipality/locality layers | `dataset/shp/final_version*.shp`, `dataset/json/final_version.json` | ~1–3 min |
| Regime comparisons | `src/analysis/boxplot.ipynb` | `dataset/shp/final_version*.shp` | Fig. 3 socio-environmental boxplots by regime | ~1 min |
| Intra-regime inequality | `src/analysis/gini.ipynb` | `dataset/shp/final_version.shp` | Table 1 Gini coefficients per index | < 1 min |
| Correlation analyses | `src/analysis/simple_corr.ipynb`, `src/analysis/multi_corr.ipynb` | Final dataset / clusters | Correlation tables | ~1 min each |
| Predictive modelling & SHAP | `src/analysis/predict.ipynb` | `dataset/shp/clusters.shp` | Water R²≈0.91, sanitation R²≈0.92, regime accuracy≈0.72; Figs. 5–7 | ~5–15 min |
| Urban scaling | `src/analysis/urban_scaling_laws.ipynb` | `dataset/shp/final_version.shp` | Scaling-law figures | ~1 min |
| Morphological signal | `src/analysis/train_bootstrap_features.py` | EfficientNetV2 embeddings (`dataset/morphology/output/`) | Per-regime F1 + 95% CI (Structured ≈ 0.883, Unstructured ≈ 0.699, Partially Structured ≈ 0.535); Table 8 | Longest step; 100 bootstrap iterations |

> **Note on imagery:** the original high-resolution satellite tiles cannot be
> redistributed due to Google Maps licensing restrictions. The morphological-signal
> demo therefore runs on the **derived EfficientNetV2 feature embeddings** provided in
> the Mendeley Data deposit, not on raw imagery.

---

## 4. Instructions for use

### Setup

1. Clone the repository and install the dependencies (Section 2).
2. Download the integrated dataset from Mendeley Data
   ([DOI 10.17632/x959s4g493.2](https://doi.org/10.17632/x959s4g493.2)) and place it
   under `dataset/`, matching the layout below.
3. Authenticate with Google Earth Engine only if you intend to re-run the environmental
   data extraction (`src/pre-processing/6-sentinel_5P.ipynb`).

### Running on your data / reproducing the manuscript

With the integrated dataset in place, run, in order:

1. **`src/cluster/generate_clusters.ipynb`** — indicator normalization, infrastructure
   index construction (Eq. 1), K-means clustering (k = 3), cluster validation (Fig. 9),
   regime assignment.
2. **`src/cluster/join_cities_data.ipynb`** — join clustered favelas with
   municipality/locality data to produce the final integrated dataset.
3. **Notebooks in `src/analysis/`** — correlation analyses, Gini coefficients, regime
   comparisons, predictive modelling (Lasso, Random Forest with RFE/SFS), SHAP, urban
   scaling.
4. **`src/analysis/train_bootstrap_features.py`** — EfficientNetV2 feature extraction,
   PCA, and bootstrapped (n = 100) SVM classification of the three regimes from
   satellite-image embeddings, with 95% confidence intervals.

### Reproduction instructions (manuscript artifact → script)

| Manuscript artifact | Produced by |
|---------------------|-------------|
| Four indices; three regimes; Fig. 9 (k selection, t-SNE); Fig. 10 (index histograms) | `src/cluster/generate_clusters.ipynb` |
| Fig. 1 (hexbin index maps); Fig. 2 (regime radar) | `src/cluster/generate_clusters.ipynb` + mapping in analysis notebooks |
| Table 1 (intra-regime Gini) | `src/analysis/gini.ipynb` |
| Fig. 3 (indicator boxplots by regime); Table 5 (regime means) | `src/analysis/boxplot.ipynb` |
| Figs. 5–6 (SHAP water/sanitation); Fig. 7 (multiclass SHAP); water/sanitation R²; regime accuracy | `src/analysis/predict.ipynb` |
| Correlation analyses | `src/analysis/simple_corr.ipynb`, `src/analysis/multi_corr.ipynb` |
| Fig. 8, Table 2 (regional distribution); Tables 6–7 (sanitation/water by region) | `src/analysis/` regional analysis + `join_cities_data.ipynb` |
| Urban scaling | `src/analysis/urban_scaling_laws.ipynb` |
| Table 8 (morphological classification F1/recall) | `src/analysis/train_bootstrap_features.py` |

### Reproducing the data harmonization (optional)

The notebooks in `src/pre-processing/` reproduce the harmonization of the raw sources
(2010/2022 Census, CIDACS IBP, AGSN 2019, MapBiomas, Sentinel-5P) into the per-favela
indicator table consumed by `generate_clusters.ipynb`. Run them in numeric order
(`1-` through `7-`). This step is **optional** because the fully harmonized dataset is
already available through Mendeley Data.

### Repository structure

```
src/
├── pre-processing/                              # Data preparation and harmonization — run in numeric order
│   ├── 1-censitary_sectors.ipynb                # Harmonizes 2010/2022 IBGE census sectors and joins census aggregates
│   ├── 2-censitary_sectors_interpolation.ipynb  # Dasymetric interpolation of the 2010 IBP onto 2022 census sectors
│   ├── 3-ibp.ipynb                              # Aggregates the Brazilian Deprivation Index (IBP, CIDACS) to favela polygons
│   ├── 4-hospitals_distances.ipynb              # Euclidean distances to AGSN 2019 primary-care and inpatient hospitals
│   ├── 5-mapbiomas.ipynb                        # Land-cover / ecological indicators (MapBiomas 2023)
│   ├── 6-sentinel_5P.ipynb                      # Air pollution (NO2, SO2, CO, O3) via Sentinel-5P / Google Earth Engine
│   └── 7-slum_variables.ipynb                   # Merges all household, within-area and area-connect indicators per favela polygon
├── cluster/
│   ├── generate_clusters.ipynb                  # Index construction + K-means (k=3)
│   └── join_cities_data.ipynb                   # Joins clustered favelas with municipality/locality data -> final integrated dataset
└── analysis/
    ├── simple_corr.ipynb                        # Bivariate correlation analyses
    ├── multi_corr.ipynb                         # Multivariate correlation analyses
    ├── boxplot.ipynb                             # Regime comparisons (Fig. 3)
    ├── gini.ipynb                                # Intra-regime Gini coefficients (Table 1)
    ├── predict.ipynb                             # Lasso / Random Forest (RFE, SFS), SHAP, regime prediction (Figs. 5–7)
    ├── urban_scaling_laws.ipynb                  # Urban scaling analyses
    └── train_bootstrap_features.py               # EfficientNetV2 + bootstrapped SVM (morphological signal, Table 8)
```

### Dataset directory

`dataset/` is **not tracked in this repository** (see `.gitignore`) because of its size.
Download it from Mendeley Data (Section 6) and place it at the repository root matching:

```
dataset/
├── favelas_urban_communities/
│   ├── fcu_blank/                 # 2022 IBGE favela / urban-community polygons (base geometry for all joins)
│   ├── agsn_2019/                 # Subnormal Agglomerations 2019 health facilities
│   │   ├── primary_care_hospitals/
│   │   └── inpatient_hospitals/
│   ├── dist_hospitals/            # Distance from each favela to nearest health facilities (output of step 4)
│   └── slum_variables/            # Raw household-level census tables, organized by domain
│       ├── population/            # Race/color, sex ratio, literacy, occupation
│       ├── household_characteristics/  # Water, sanitation, bathrooms, waste destination
│       ├── establishments/        # Local economic-activity counts
│       ├── area_density.xlsx      # Household/population density inputs
│       └── final/                 # Merged household + within-area + area-connect indicators (output of step 7)
├── census_sectors/                # 2022 IBGE census sectors + aggregated indicators (output of steps 1–2)
├── census_sectors_2010/           # 2010 IBGE census sectors (used to interpolate the 2010 IBP onto 2022 sectors)
├── ibp/                           # Brazilian Deprivation Index, CIDACS (output of step 3)
├── air_pollution/                 # Sentinel-5P favela polygons + results/ (one NO2/SO2/CO/O3 CSV per favela, output of step 6)
├── mapbiomas/                     # MapBiomas 2023 land cover intersected with favela polygons (output of step 5)
├── br_locales_2010/                # IBGE 2010 localities (urban/rural reference, used in join_cities_data.ipynb)
├── br_municipalities_2022/        # IBGE 2022 municipality boundaries (used in join_cities_data.ipynb)
├── clusters/                      # Favela polygons with the 4 indices and regime label (output of generate_clusters.ipynb)
├── shp/                           # Final integrated dataset used by the analysis notebooks
│   ├── final_version.shp          # All favelas with indicators, indices and regime labels
│   └── final_version_no_missing_values.shp  # Same, restricted to complete cases
├── morphology/                    # EfficientNetV2 embeddings for the morphological-signal analysis
│   └── output/                    # Derived image feature embeddings (no raw imagery — see licensing note)
└── json/
    └── final_version.json         # GeoJSON export of the final integrated dataset
```

---

## 5. Reproducibility

- **Clustering:** K-means with `k=3`, 10 random initializations, fixed
  `random_state=42` (`src/cluster/generate_clusters.ipynb`).
- **Predictive modelling:** Lasso, Random Forest (RFE and SFS variants) trained with
  10-fold cross-validation (`src/analysis/predict.ipynb`).
- **Morphological signal:** bootstrapped evaluation with n = 100 iterations and a fixed
  base seed (`src/analysis/train_bootstrap_features.py`), from which mean F1-scores and
  95% confidence intervals are computed per regime.

Running the workflow on the dataset provided through Mendeley Data should reproduce the
reported results within expected numerical precision. Because some steps involve
stochastic resampling (K-means initializations, bootstrap subsampling), minor numerical
variation may occur across environments even with fixed seeds (e.g., due to library
version differences); this does not affect the qualitative conclusions.

---

## 6. Data availability

The minimum dataset required to interpret, verify, and extend the findings of this study
is publicly available through **Mendeley Data**:

> **DOI: [10.17632/x959s4g493.2](https://doi.org/10.17632/x959s4g493.2)**

The Mendeley Data record includes:

- Favela polygon shapefile with all integrated indicators and regime labels
  (equivalent to `dataset/shp/` and `dataset/json/`, see Section 4)
- Raw and intermediate source layers used in `src/pre-processing/`
- Data dictionary
- EfficientNetV2 image feature embeddings used in the morphological analysis
  (`dataset/morphology/output/`)

**Third-party data restriction:** the original high-resolution satellite imagery cannot
be redistributed due to Google Maps licensing restrictions. Only the derived image
embeddings used in the experiments are provided.

Primary data sources:

| Scale | Domain | Sources |
|---------|---------|---------|
| Household | Census and socio-economic conditions | 2022 Census (IBGE), Brazilian Deprivation Index / IBP (CIDACS) |
| Within-area | Environmental conditions | MapBiomas 2023, Sentinel-5P (via Google Earth Engine), digital elevation models (slope, HAND) |
| Area-connect | Accessibility and infrastructure | Subnormal Agglomerations 2019 (AGSN, IBGE), health facility datasets |

All datasets were harmonized to the 2022 Census favela polygons (12,344 settlements
nationwide) using the SIRGAS 2000 coordinate reference system.

---

## 7. Code availability

All custom code central to the conclusions of this study is openly available:

- **Source repository (GitHub):**
  https://github.com/dev-jotape/state_of_brazilian_favelas
- **Archived, citable deposit (DOI-minting repository):** Mendeley Data,
  [DOI 10.17632/x959s4g493.2](https://doi.org/10.17632/x959s4g493.2)

The code is released under the **Apache License 2.0** (Section 8). Custom code central to
the manuscript includes: infrastructure and housing index construction, K-means
clustering, Gini-coefficient computation, Lasso/Random Forest predictive modelling with
SHAP explanations, and the bootstrapped EfficientNetV2 + SVM morphological-signal
analysis.

---

## 8. License

### Code

Released under the **Apache License 2.0**. See the [`LICENSE`](LICENSE) file.

### Dataset

Released under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**,
as stated in the [`LICENSE`](LICENSE) file and in the Mendeley Data record.

---

## 9. Citation

> Silva, J.P., Porto de Albuquerque, J., Pedrassoli, J.C., Spadon, G.,
> Rodrigues-Jr, J.F. *The State of Brazilian Favelas: Infrastructural Regimes and
> Urban Inequality* (2026).

**Dataset:**
> da Silva, João Pedro (2026), "The State of Brazilian Favelas", Mendeley Data, V2,
> doi: [10.17632/x959s4g493.2](https://doi.org/10.17632/x959s4g493.2)

## 10. Authors

- **João Pedro da Silva** (jp.silva@usp.br) — ICMC, University of São Paulo
- João Porto de Albuquerque — School of Social and Political Sciences, University of Glasgow
- Julio Cesar Pedrassoli — Department of Geography, University of São Paulo
- Gabriel Spadon — Faculty of Computer Science, Dalhousie University
- Jose F. Rodrigues-Jr — ICMC, University of São Paulo
