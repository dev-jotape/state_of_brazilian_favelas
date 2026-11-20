"""R3.31 - spatially-blocked cross-validation for the water, sanitation and regime models.
Features = the RFECV-selected set of the baseline run (01_models_baseline.json), RF(n_estimators=200, random_state=42).
Compared schemes (5 folds each unless noted): random (KFold / StratifiedKFold shuffle) vs GroupKFold by municipality,
GroupKFold by state (UF), leave-one-macro-region-out (5 folds). Metrics are pooled out-of-fold, plus per-fold mean.
Optional: --nested also re-runs RFECV inside each leave-one-region-out training fold (feature selection without leakage)."""
import sys, json, numpy as np, pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, LeaveOneGroupOut
from pred_common import *

base = json.load(open(f'{OUT}/01_models_baseline.json'))
df = load_pred()
nested = '--nested' in sys.argv
Xall = pd.DataFrame(StandardScaler().fit_transform(df[X_VARS]), columns=X_VARS)
groups = {'municipality': df['CD_MUN'].values, 'state(UF)': df['CD_UF'].values, 'macro-region': df['region'].values}
rows = []

def run_cv(kind, tgt, feats, splitter, groups_=None):
    y = df[tgt].values; oof = np.zeros(len(df), dtype=float if kind != 'regime' else int); per = []
    for tr, te in splitter.split(Xall, y, groups_):
        m = make_rf(kind).fit(Xall.iloc[tr][feats], y[tr]); oof[te] = m.predict(Xall.iloc[te][feats])
        per.append(metric(kind, y[te], oof[te]))
    pooled = metric(kind, y, oof)
    return pooled, {k: float(np.mean([p[k] for p in per])) for k in per[0]}, per

for kind, tgt in TARGETS.items():
    feats = base[kind]['selected']
    schemes = {'random 5-fold (current-style)': (StratifiedKFold(5, shuffle=True, random_state=42) if kind == 'regime' else KFold(5, shuffle=True, random_state=42), None),
               'GroupKFold municipality': (GroupKFold(5), groups['municipality']),
               'GroupKFold state': (GroupKFold(5), groups['state(UF)']),
               'Leave-one-macro-region-out': (LeaveOneGroupOut(), groups['macro-region'])}
    for name, (sp, gr) in schemes.items():
        pooled, mean_fold, per = run_cv(kind, tgt, feats, sp, gr)
        rows.append({'target': kind, 'scheme': name, 'features': 'baseline RFECV set', **{f'pooled_{k}': v for k, v in pooled.items()},
                     **{f'foldmean_{k}': v for k, v in mean_fold.items()}, 'n_folds': len(per),
                     'per_fold': json.dumps([{k: round(v, 4) for k, v in p.items()} for p in per])})
        print(rows[-1]['target'], name, pooled, flush=True)
    if nested:
        y = df[tgt].values; oof = np.zeros(len(df), dtype=float if kind != 'regime' else int); per = []
        for tr, te in LeaveOneGroupOut().split(Xall, y, groups['macro-region']):
            _, f = run_rfecv(kind, Xall.iloc[tr].reset_index(drop=True), y[tr])
            m = make_rf(kind).fit(Xall.iloc[tr][f], y[tr]); oof[te] = m.predict(Xall.iloc[te][f]); per.append(metric(kind, y[te], oof[te]))
        pooled = metric(kind, y, oof)
        rows.append({'target': kind, 'scheme': 'LORO + RFECV inside each training fold', 'features': 'nested RFECV', **{f'pooled_{k}': v for k, v in pooled.items()},
                     **{f'foldmean_{k}': float(np.mean([p[k] for p in per])) for k in per[0]}, 'n_folds': len(per),
                     'per_fold': json.dumps([{k: round(v, 4) for k, v in p.items()} for p in per])})
        print(rows[-1]['target'], 'nested', pooled, flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/R3_31_spatial_cv.csv', index=False)
# LORO per held-out region detail (baseline features)
det = []
for kind, tgt in TARGETS.items():
    feats = base[kind]['selected']; y = df[tgt].values
    for R in REGION.values():
        te = (df['region'] == R).values; tr = ~te
        m = make_rf(kind).fit(Xall[tr][feats], y[tr]); det.append({'target': kind, 'held_out_region': R, 'n': int(te.sum()), **metric(kind, y[te], m.predict(Xall[te][feats]))})
pd.DataFrame(det).to_csv(f'{OUT}/R3_31_loro_per_region.csv', index=False); print(pd.DataFrame(det).round(4).to_string())
