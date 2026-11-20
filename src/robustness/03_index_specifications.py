"""R3.21 (+R1.24) - alternative index specifications vs the current solution."""
import numpy as np, pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from common import *

g = load_slum()
ref_X = build_indices(g)
ref = kmeans_labels(ref_X)
rows = []
for name, kw in {'Exponential, data min-max (paper)': dict(weights='exp', norm='data'),
                 'Linear weights, data min-max': dict(weights='linear', norm='data'),
                 'Equal weights (best-category shares), data min-max': dict(weights='equal', norm='data'),
                 'Exponential, theoretical-interval min-max': dict(weights='exp', norm='theory'),
                 'Linear weights, theoretical-interval min-max': dict(weights='linear', norm='theory'),
                 'Equal weights, theoretical-interval min-max': dict(weights='equal', norm='theory')}.items():
    X = build_indices(g, **kw)
    lab = kmeans_labels(X)
    sizes = pd.Series(paper_labels(lab) if kw == dict(weights='exp', norm='data') else lab).value_counts().sort_values(ascending=False)
    r = {'specification': name, 'silhouette_k3': silhouette_score(X, lab), 'davies_bouldin': davies_bouldin_score(X, lab),
         'ARI_vs_paper': adjusted_rand_score(ref, lab), 'NMI_vs_paper': normalized_mutual_info_score(ref, lab),
         'sizes_pct(desc)': '/'.join(f'{v:.1f}' for v in sizes / len(lab) * 100),
         'best_k_by_silhouette(2-5)': int(max(range(2, 6), key=lambda k: silhouette_score(X, kmeans_labels(X, k))))}
    r.update({f'corr_{c}_vs_paper': float(np.corrcoef(X[c], ref_X[c])[0, 1]) for c in IDX})
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/R3_21_index_specifications.csv', index=False)
print(df.round(4).T.to_string())
