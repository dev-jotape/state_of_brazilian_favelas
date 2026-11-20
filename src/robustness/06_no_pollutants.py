"""R3.32 - compare models with and without Sentinel-5P pollutants (needs 01_models_baseline.json and 01_models_nopoll.json,
the latter produced by `python 01_baseline_models.py --no-poll`). Also SHAP mean|value| shares for both settings."""
import json, numpy as np, pandas as pd, shap
from pred_common import *
df = load_pred(); rows, shap_rows = [], []
for tag in ('baseline', 'nopoll'):
    res = json.load(open(f'{OUT}/01_models_{tag}.json'))
    feats_all = [f for f in X_VARS if not (tag == 'nopoll' and f in POLLUTANTS)]
    for kind, tgt in TARGETS.items():
        r = res[kind]; feats = r['selected']
        Xtr, Xte, ytr, yte, tr, te = split(df, feats_all, tgt)
        m = make_rf(kind).fit(Xtr[feats], ytr)
        sv = shap.TreeExplainer(m).shap_values(Xte[feats])
        sv = np.stack(sv, -1) if isinstance(sv, list) else sv
        ma = np.abs(sv).mean(axis=tuple(i for i in range(sv.ndim) if i != 1))
        share = 100 * ma / ma.sum()
        for f, s, a in zip(feats, share, ma): shap_rows.append({'setting': tag, 'target': kind, 'feature': f, 'mean_abs_shap': a, 'shap_share_pct': s})
        rows.append({'setting': 'with pollutants' if tag == 'baseline' else 'without pollutants', 'target': kind, 'n_candidate_features': len(feats_all),
                     'n_selected': len(feats), **{f'test_{k}': v for k, v in r['test'].items()}, 'rfecv_cv_best': r['rfecv_cv_best'],
                     'pollutants_selected': ','.join(f for f in feats if f in POLLUTANTS), 'selected': ','.join(feats)})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/R3_32_pollutants_comparison.csv', index=False)
sh = pd.DataFrame(shap_rows); sh.to_csv(f'{OUT}/R3_32_shap_importance.csv', index=False)
for kind in TARGETS:
    p = sh[sh.target == kind].pivot(index='feature', columns='setting', values='shap_share_pct').sort_values('nopoll', ascending=False)
    print(kind); print(p.round(2).to_string())
