"""R3.19 - sensitivity of the k=3 regimes to the clustering algorithm."""
import numpy as np, pandas as pd, json, time
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score, calinski_harabasz_score
from sklearn.metrics.pairwise import euclidean_distances
from common import *

def kmedoids(X, k=3, seed=42, n_init=5, max_iter=100):
    """Voronoi-iteration k-medoids (Euclidean) on the full distance matrix (k-medoids++ style init)."""
    D = euclidean_distances(X).astype(np.float32)
    rng = np.random.RandomState(seed); best = (np.inf, None)
    for _ in range(n_init):
        med = [rng.randint(len(X))]
        for _ in range(k - 1):
            d = D[:, med].min(1) ** 2
            med.append(rng.choice(len(X), p=d / d.sum()))
        med = np.array(med)
        for _ in range(max_iter):
            lab = D[:, med].argmin(1)
            new = med.copy()
            for c in range(k):
                idx = np.where(lab == c)[0]
                new[c] = idx[D[np.ix_(idx, idx)].sum(1).argmin()]
            if (new == med).all(): break
            med = new
        cost = D[:, med].min(1).sum()
        if cost < best[0]: best = (cost, D[:, med].argmin(1))
    return best[1]

g = load_slum()
X = build_indices(g).values
ref = paper_labels(kmeans_labels(X))
methods = {
    'KMeans (paper)': lambda: kmeans_labels(X),
    'MiniBatchKMeans': lambda: MiniBatchKMeans(3, random_state=42, n_init=10, batch_size=2048).fit_predict(X),
    'GMM full': lambda: GaussianMixture(3, covariance_type='full', random_state=42, n_init=10).fit_predict(X),
    'GMM diag': lambda: GaussianMixture(3, covariance_type='diag', random_state=42, n_init=10).fit_predict(X),
    'GMM spherical': lambda: GaussianMixture(3, covariance_type='spherical', random_state=42, n_init=10).fit_predict(X),
    'Agglomerative Ward': lambda: AgglomerativeClustering(3, linkage='ward').fit_predict(X),
    'Agglomerative average': lambda: AgglomerativeClustering(3, linkage='average').fit_predict(X),
    'k-medoids (PAM-style)': lambda: kmedoids(X),
}
rows, labs = [], {}
for name, f in methods.items():
    t = time.time(); lab = f(); labs[name] = lab
    sizes = pd.Series(lab).value_counts().sort_values(ascending=False)
    rows.append({'method': name, 'silhouette': silhouette_score(X, lab), 'davies_bouldin': davies_bouldin_score(X, lab),
                 'calinski_harabasz': calinski_harabasz_score(X, lab),
                 'ARI_vs_paper': adjusted_rand_score(ref, lab), 'NMI_vs_paper': normalized_mutual_info_score(ref, lab),
                 'sizes_pct(desc)': '/'.join(f'{v:.1f}' for v in (sizes / len(lab) * 100)), 'sizes_n': '/'.join(map(str, sizes.values)),
                 'secs': round(time.time() - t, 1)})
    print(rows[-1], flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/R3_19_cluster_sensitivity.csv', index=False)
# k-selection per method (silhouette k=2..5) for the cheap ones
ks = []
for k in range(2, 6):
    for name, f in {'KMeans': lambda: kmeans_labels(X, k),
                    'GMM full': lambda: GaussianMixture(k, covariance_type='full', random_state=42, n_init=10).fit_predict(X),
                    'Ward': lambda: AgglomerativeClustering(k, linkage='ward').fit_predict(X)}.items():
        lab = f(); ks.append({'k': k, 'method': name, 'silhouette': silhouette_score(X, lab), 'davies_bouldin': davies_bouldin_score(X, lab)})
pd.DataFrame(ks).to_csv(f'{OUT}/R3_19_k_selection_by_method.csv', index=False)
print(pd.DataFrame(ks).pivot(index='k', columns='method', values='silhouette').round(3))
print(df.round(4).to_string())
