"""Reproduce README predictive numbers (water R2~0.91, sanitation R2~0.92, regime acc~0.72) and store the selected features.
predict.ipynb in the repo only contains the regime classifier; water/sanitation are the same recipe with
RandomForestRegressor + RFECV(scoring='r2') (reconstructed here). Optional arg: --no-poll (R3.32)."""
import sys, json, time, joblib
from pred_common import *
no_poll = '--no-poll' in sys.argv
tag = 'nopoll' if no_poll else 'baseline'
feats_all = [f for f in X_VARS if not (no_poll and f in POLLUTANTS)]
df = load_pred()
print(len(df), 'rows after dropna')
res = {}
for kind, tgt in TARGETS.items():
    t = time.time()
    Xtr, Xte, ytr, yte, tr, te = split(df, feats_all, tgt)
    r, feats = run_rfecv(kind, Xtr, ytr)
    m = make_rf(kind).fit(Xtr[feats], ytr)
    p = m.predict(Xte[feats])
    res[kind] = {'n_train': len(tr), 'n_test': len(te), 'n_features_all': len(feats_all), 'n_selected': len(feats), 'selected': feats,
                 'test': metric(kind, yte, p), 'rfecv_cv_best': float(np.max(r.cv_results_['mean_test_score'])),
                 'importances': dict(zip(feats, map(float, m.feature_importances_)))}
    print(kind, res[kind]['test'], len(feats), f'{time.time()-t:.0f}s', flush=True)
json.dump(res, open(f'{OUT}/01_models_{tag}.json', 'w'), indent=1)
