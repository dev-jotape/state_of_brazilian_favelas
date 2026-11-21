"""Runs the RFECV selection for water, sanitation, regime, with and without the Sentinel-5P pollutants. Output: 01_selection.json (incremental)."""
import json, os, time
from pred_common import *
df = load_pred(); path = f'{OUT}/01_selection.json'
res = json.load(open(path)) if os.path.exists(path) else {}
for setting in ('with_pollutants', 'without_pollutants'):
    feats_all = [f for f in X_VARS if not (setting == 'without_pollutants' and f in POLLUTANTS)]
    for kind in ('regime', 'water', 'sanitation'):
        key = f'{setting}/{kind}'
        if key in res: continue
        t = time.time(); feats, imp, n, best = select_features(df, kind, feats_all)
        res[key] = {'n_candidates': len(feats_all), 'n_rows_used_in_rfecv': n, 'selected': feats, 'rf_importance': imp, 'rfecv_best_cv_score': best}
        json.dump(res, open(path, 'w'), indent=1)
        print(key, len(feats), f'best CV {best:.4f}', f'{time.time()-t:.0f}s', flush=True)
