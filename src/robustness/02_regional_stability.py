"""R3.20 - stability of the regimes across macro-regions (per-region clustering + leave-one-region-out)."""
import numpy as np, pandas as pd, json
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from common import *

g = load_slum()
X = build_indices(g)
nat = paper_labels(kmeans_labels(X))
reg = regions(g.CD_FCU)
nat_cent = X.groupby(nat).mean().loc[[0, 1, 2]]
names = {0: 'Structured', 1: 'Partially', 2: 'Unstructured'}

def match(ref_cent, cent):
    """map cluster ids of `cent` to national regime ids by minimising centroid distance."""
    D = np.linalg.norm(ref_cent.values[:, None, :] - cent.values[None, :, :], axis=2)
    r, c = linear_sum_assignment(D)
    return {cent.index[j]: ref_cent.index[i] for i, j in zip(r, c)}, D[r, c]

rows, cents, cms = [], [], {}
# (a) clustering run separately within each macro-region (national indices, k=3, same hyper-parameters)
for R in REGION.values():
    m = reg == R; Xr = X[m]; nr = nat[m]
    lab = kmeans_labels(Xr)
    cent = Xr.groupby(lab).mean()
    mp, dist = match(nat_cent, cent)
    lab_m = pd.Series(lab).map(mp).values
    cm = confusion_matrix(nr, lab_m, labels=[0, 1, 2])
    cms[R] = cm
    rows.append({'region': R, 'n': int(m.sum()), 'silhouette_regional_k3': silhouette_score(Xr, lab),
                 'ARI_vs_national': adjusted_rand_score(nr, lab), 'NMI_vs_national': normalized_mutual_info_score(nr, lab),
                 'agreement_after_matching': float((nr == lab_m).mean()),
                 'national_regime_pct': '/'.join(f'{(nr == c).mean()*100:.1f}' for c in range(3)),
                 'regional_cluster_pct(matched)': '/'.join(f'{(lab_m == c).mean()*100:.1f}' for c in range(3)),
                 'mean_centroid_shift_vs_national': float(dist.mean()), 'max_centroid_shift_vs_national': float(dist.max())})
    for old, new in mp.items():
        c = cent.loc[old]; cents.append({'region': R, 'regime(matched)': names[new], **c.round(4).to_dict()})
for c in range(3):
    cents.append({'region': 'National', 'regime(matched)': names[c], **nat_cent.loc[c].round(4).to_dict()})
per = pd.DataFrame(rows); per.to_csv(f'{OUT}/R3_20_per_region.csv', index=False)
pd.DataFrame(cents).to_csv(f'{OUT}/R3_20_centroids_by_region.csv', index=False)
with open(f'{OUT}/R3_20_confusion_matrices.txt', 'w') as f:
    for R, cm in cms.items():
        f.write(f'{R}: rows = national regime (Structured, Partially, Unstructured), cols = regional cluster matched to the national regimes\n{cm}\n\n')
# (b) leave-one-region-out: fit k-means (k=3) without region R, assign R to nearest centroid
lo = []
for R in REGION.values():
    m = reg == R
    km = KMeans(3, random_state=42, n_init=10).fit(X[~m])
    cent = pd.DataFrame(km.cluster_centers_, columns=IDX)
    mp, dist = match(nat_cent, cent)
    pr = pd.Series(km.predict(X[m])).map(mp).values
    lo.append({'held_out': R, 'n_heldout': int(m.sum()), 'ARI_vs_national': adjusted_rand_score(nat[m], pr),
               'accuracy_vs_national': float((nat[m] == pr).mean()), 'max_centroid_shift': float(dist.max()),
               'heldout_pct_pred': '/'.join(f'{(pr == c).mean()*100:.1f}' for c in range(3)),
               'heldout_pct_national': '/'.join(f'{(nat[m] == c).mean()*100:.1f}' for c in range(3))})
lo = pd.DataFrame(lo); lo.to_csv(f'{OUT}/R3_20_leave_one_region_out.csv', index=False)
print(per.round(4).to_string()); print(lo.round(4).to_string()); print(pd.DataFrame(cents).to_string())
for R, cm in cms.items(): print(R, cm.tolist())
