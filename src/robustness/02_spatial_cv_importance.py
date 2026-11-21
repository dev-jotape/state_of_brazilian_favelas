"""R3.31 - importance rankings under random vs spatially blocked CV (water, sanitation, regime).
Predictors = features selected by RFECV with the original recipe (01_selection.json, with_pollutants). In every fold an RF with the
recipe's hyper-parameters is fit on the training folds; permutation importance (n_repeats=5, scoring r2 / accuracy) is computed on the held-out
fold, RF impurity importance on the training fold; both are averaged over folds. Schemes: random (KFold/StratifiedKFold shuffle, seed 42),
GroupKFold by municipality (CD_MUN), GroupKFold by state (CD_UF), leave-one-macro-region-out (5 folds)."""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, LeaveOneGroupOut
from sklearn.inspection import permutation_importance
from pred_common import *

sel = json.load(open(f'{OUT}/01_selection.json'))
df = load_pred()
Xall = scaled(df, X_VARS)
G = {'municipality': df['CD_MUN'].values, 'state': df['CD_UF'].values, 'macro-region': df['region'].values}
long, perf, loro_rows = [], [], []

for kind, tgt in TARGETS.items():
    feats = sel[f'with_pollutants/{kind}']['selected']; y = df[tgt].values; X = Xall[feats]
    scoring = 'accuracy' if kind == 'regime' else 'r2'
    schemes = {'random': (StratifiedKFold(5, shuffle=True, random_state=42) if kind == 'regime' else KFold(5, shuffle=True, random_state=42), None),
               'GroupKFold municipality': (GroupKFold(5), G['municipality']),
               'GroupKFold state': (GroupKFold(5), G['state']),
               'Leave-one-macro-region-out': (LeaveOneGroupOut(), G['macro-region'])}
    for sname, (sp, gr) in schemes.items():
        perm, rfi, oof = [], [], np.zeros(len(df))
        for tr, te in sp.split(X, y, gr):
            m = make_rf(kind).fit(X.iloc[tr], y[tr])
            oof[te] = m.predict(X.iloc[te])
            pi = permutation_importance(m, X.iloc[te], y[te], scoring=scoring, n_repeats=5, random_state=42, n_jobs=2)
            perm.append(pi.importances_mean); rfi.append(m.feature_importances_)
            if sname.startswith('Leave'):
                loro_rows += [{'model': kind, 'held_out': np.unique(gr[te])[0], 'feature': f, 'perm_importance': v, 'rf_importance': r}
                              for f, v, r in zip(feats, pi.importances_mean, m.feature_importances_)]
        perm = np.mean(perm, 0); rfi = np.mean(rfi, 0)
        pos = np.clip(perm, 0, None)
        perf.append({'model': kind, 'scheme': sname, 'n_folds': sp.get_n_splits(X, y, gr),
                     'pooled_oof_' + scoring: (r2_score(y, oof) if kind != 'regime' else accuracy_score(y, oof.astype(int)))})
        for i, f in enumerate(feats):
            long.append({'model': kind, 'scheme': sname, 'feature': f, 'rf_importance_pct': 100 * rfi[i] / rfi.sum(),
                         'perm_importance': perm[i], 'perm_importance_pct': 100 * pos[i] / pos.sum()})
        print(kind, sname, perf[-1], flush=True)
L = pd.DataFrame(long)
L['rank_rf'] = L.groupby(['model', 'scheme']).rf_importance_pct.rank(ascending=False, method='min').astype(int)
L['rank_perm'] = L.groupby(['model', 'scheme']).perm_importance.rank(ascending=False, method='min').astype(int)
L.to_csv(f'{OUT}/R3_31_importance_by_scheme.csv', index=False)
pd.DataFrame(perf).to_csv(f'{OUT}/R3_31_oof_performance_context.csv', index=False)
pd.DataFrame(loro_rows).to_csv(f'{OUT}/R3_31_loro_importance_per_heldout_region.csv', index=False)
# random vs spatial agreement
cmp = []
for kind in TARGETS:
    for sname in ['GroupKFold municipality', 'GroupKFold state', 'Leave-one-macro-region-out']:
        a = L[(L.model == kind) & (L.scheme == 'random')].set_index('feature'); b = L[(L.model == kind) & (L.scheme == sname)].set_index('feature').loc[a.index]
        n = len(a)
        for metric, col in [('permutation', 'perm_importance'), ('RF impurity', 'rf_importance_pct')]:
            top = lambda s, k: set(s.sort_values(ascending=False).index[:min(k, n)])
            cmp.append({'model': kind, 'n_selected': n, 'spatial_scheme': sname, 'importance': metric,
                        'spearman': spearmanr(a[col], b[col])[0], 'top5_overlap': len(top(a[col], 5) & top(b[col], 5)), 'top10_overlap': f'{len(top(a[col], 10) & top(b[col], 10))}/{min(10, n)}'})
C = pd.DataFrame(cmp); C.to_csv(f'{OUT}/R3_31_random_vs_spatial_agreement.csv', index=False)
print(C.round(3).to_string())
for kind in TARGETS:
    for col in ['perm_importance_pct', 'rf_importance_pct']:
        print(kind, col); print(L[L.model == kind].pivot(index='feature', columns='scheme', values=col).round(2).sort_values('random', ascending=False).to_string())
