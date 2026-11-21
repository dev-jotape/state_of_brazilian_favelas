"""Sanity check: do the indices/regimes rebuilt here reproduce the stored clusters.shp and README numbers?"""
import numpy as np, pandas as pd, geopandas as gpd, json
from sklearn.metrics import silhouette_score, adjusted_rand_score
from common import *

g = load_slum()
X = build_indices(g)
lab = kmeans_labels(X)
paper = paper_labels(lab)
stored = gpd.read_file(CLUSTERS, ignore_geometry=True)
m = g[['CD_FCU']].assign(lab=paper).merge(stored[['CD_FCU', 'wtr_idx', 'sanit_idx', 'living_are', 'durable_ho', 'cluster_3']], on='CD_FCU')
res = {
    'n': len(g),
    'silhouette_k3': float(silhouette_score(X.values, lab)),
    'regime_pct': (pd.Series(paper).value_counts(normalize=True).sort_index() * 100).round(2).to_dict(),
    'ARI_vs_stored_clusters_shp': float(adjusted_rand_score(m.lab, m.cluster_3)),
    'identical_labels_vs_stored': float((m.lab == m.cluster_3).mean()),
    'max_abs_diff_idx': {c: float(np.abs(X[c].values - m[s].values).max()) for c, s in zip(IDX, ['wtr_idx', 'sanit_idx', 'living_are', 'durable_ho'])},
}
for k in range(2, 6):
    res[f'silhouette_k{k}'] = float(silhouette_score(X.values, kmeans_labels(X, k)))
print(json.dumps(res, indent=1))
json.dump(res, open(f'{OUT}/00_baseline_reproduction.json', 'w'), indent=1)
