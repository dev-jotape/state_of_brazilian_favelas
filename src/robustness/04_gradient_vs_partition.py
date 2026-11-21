"""R3.22/R3.23 - are the regimes a partition of a water-sanitation gradient? subset clustering, ANOVA eta^2, RF importance."""
import itertools, numpy as np, pandas as pd, json
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, accuracy_score, f1_score
from sklearn.decomposition import PCA
from common import *

g = load_slum(); X = build_indices(g)
ref = paper_labels(kmeans_labels(X))
short = {'wtr_idx': 'W', 'sanit_idx': 'S', 'living_area_idx': 'L', 'durable_house_idx': 'D'}
# --- descriptive: dispersion and redundancy of the four indices
desc = pd.DataFrame({'mean': X.mean(), 'std': X.std(), 'IQR': X.quantile(.75) - X.quantile(.25),
                     'pct_at_max(>=0.999)': (X >= 0.999).mean() * 100, 'pct_at_min(<=0.001)': (X <= 0.001).mean() * 100})
desc.to_csv(f'{OUT}/R3_22_index_descriptives.csv'); X.corr().to_csv(f'{OUT}/R3_22_index_correlations.csv'); X.corr(method='spearman').to_csv(f'{OUT}/R3_22_index_correlations_spearman.csv')
pca = PCA().fit(X); pcadf = pd.DataFrame(pca.components_, columns=IDX); pcadf.insert(0, 'explained_var_ratio', pca.explained_variance_ratio_)
pcadf.to_csv(f'{OUT}/R3_22_pca_of_indices.csv')
print(desc.round(3)); print(X.corr().round(3)); print(pcadf.round(3))
# --- eta^2 (one-way ANOVA of each index on the regime)
eta = []
for c in IDX:
    grand = X[c].mean(); ssb = sum(((X[c][ref == k].mean() - grand) ** 2) * (ref == k).sum() for k in range(3)); sst = ((X[c] - grand) ** 2).sum()
    F, p = stats.f_oneway(*[X[c][ref == k] for k in range(3)])
    eta.append({'index': c, 'eta_squared': ssb / sst, 'F': F, 'p': p})
eta = pd.DataFrame(eta); eta.to_csv(f'{OUT}/R3_22_eta_squared.csv', index=False); print(eta)
# --- clustering on subsets of indices
rows = []
for r in (4, 3, 2):
    for sub in itertools.combinations(IDX, r):
        Xs = X[list(sub)]; lab = kmeans_labels(Xs)
        rows.append({'indices': '+'.join(short[s] for s in sub), 'n_indices': r, 'silhouette': silhouette_score(Xs, lab),
                     'ARI_vs_4index_regimes': adjusted_rand_score(ref, lab), 'NMI_vs_4index_regimes': normalized_mutual_info_score(ref, lab),
                     'sizes_pct(desc)': '/'.join(f'{v:.1f}' for v in pd.Series(lab).value_counts().sort_values(ascending=False) / len(lab) * 100)})
cl = pd.DataFrame(rows); cl.to_csv(f'{OUT}/R3_22_subset_clustering.csv', index=False); print(cl.round(4).to_string())
# --- RF predicting the 4-index regime from subsets of indices (5-fold stratified CV, out-of-fold)
rf_rows = []
for r in (1, 2, 3, 4):
    for sub in itertools.combinations(IDX, r):
        p = cross_val_predict(RandomForestClassifier(200, random_state=42, n_jobs=-1), X[list(sub)], ref, cv=StratifiedKFold(5, shuffle=True, random_state=42))
        rf_rows.append({'predictors': '+'.join(short[s] for s in sub), 'accuracy': accuracy_score(ref, p), 'macro_f1': f1_score(ref, p, average='macro'),
                        'ARI': adjusted_rand_score(ref, p)})
rf = pd.DataFrame(rf_rows); rf.to_csv(f'{OUT}/R3_22_rf_regime_from_index_subsets.csv', index=False); print(rf.round(4).to_string())
# --- permutation-free importance from the full RF + k-means centroid spread (between-cluster separation per index)
m = RandomForestClassifier(500, random_state=42, n_jobs=-1).fit(X, ref)
imp = pd.DataFrame({'index': IDX, 'rf_gini_importance': m.feature_importances_, 'centroid_range(max-min over regimes)': [X[c].groupby(ref).mean().max() - X[c].groupby(ref).mean().min() for c in IDX]})
imp.to_csv(f'{OUT}/R3_22_rf_importance_and_centroid_range.csv', index=False); print(imp)
X.groupby(ref).mean().to_csv(f'{OUT}/R3_22_regime_centroids.csv')
