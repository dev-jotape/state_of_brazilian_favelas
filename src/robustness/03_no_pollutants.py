"""R3.32 - same recipe without SO2, CO, O3, NO2: selected variables, importance ranking and change of the remaining variables.
Importances: RF impurity importance of the final RFECV estimator (as the notebook) and mean|SHAP| share (TreeExplainer; regressors on a random 2000-row sample of the rows used for fitting
(deviation from multi_corr.ipynb, which uses all rows; too slow here), regime on the 20% hold-out as in predict.ipynb)."""
import json, shap, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from pred_common import *
sel = json.load(open(f'{OUT}/01_selection.json')); df = load_pred()
rows = []
for setting in ('with_pollutants', 'without_pollutants'):
    feats_all = [f for f in X_VARS if not (setting == 'without_pollutants' and f in POLLUTANTS)]
    x = scaled(df, feats_all)
    for kind, tgt in TARGETS.items():
        feats = sel[f'{setting}/{kind}']['selected']; y = df[tgt].values
        if kind == 'regime':
            tr, te = train_test_split(np.arange(len(df)), stratify=y, test_size=0.2, random_state=42)
            m = make_rf(kind).fit(x.iloc[tr][feats], y[tr]); Xs = x.iloc[te][feats]
        else:
            m = make_rf(kind).fit(x[feats], y); Xs = x[feats].sample(2000, random_state=42)  # DEVIATION: notebook explains all rows; subsample for speed (full-depth RF TreeSHAP >1h)
        sv = shap.TreeExplainer(m).shap_values(Xs); sv = np.stack(sv, -1) if isinstance(sv, list) else sv
        ma = np.abs(sv).mean(axis=tuple(i for i in range(sv.ndim) if i != 1))
        for f, a, r in zip(feats, ma, m.feature_importances_):
            rows.append({'setting': setting, 'model': kind, 'feature': f, 'rf_importance_pct': 100 * r, 'shap_pct': 100 * a / ma.sum()})
L = pd.DataFrame(rows)
L['rank_rf'] = L.groupby(['setting', 'model']).rf_importance_pct.rank(ascending=False, method='min').astype(int)
L['rank_shap'] = L.groupby(['setting', 'model']).shap_pct.rank(ascending=False, method='min').astype(int)
L.to_csv(f'{OUT}/R3_32_importance_with_vs_without_pollutants.csv', index=False)
summ = []
for kind in TARGETS:
    a = L[(L.setting == 'with_pollutants') & (L.model == kind)].set_index('feature'); b = L[(L.setting == 'without_pollutants') & (L.model == kind)].set_index('feature')
    common = a.index.intersection(b.index)
    summ.append({'model': kind, 'n_selected_with': len(a), 'n_selected_without': len(b),
                 'pollutants_selected_with(rf%,shap%)': '; '.join(f'{f} ({a.loc[f,"rf_importance_pct"]:.1f},{a.loc[f,"shap_pct"]:.1f})' for f in a.index if f in POLLUTANTS),
                 'pollutants_total_rf_pct': sum(a.loc[f, 'rf_importance_pct'] for f in a.index if f in POLLUTANTS),
                 'pollutants_total_shap_pct': sum(a.loc[f, 'shap_pct'] for f in a.index if f in POLLUTANTS),
                 'n_common': len(common), 'only_with': ','.join(sorted(set(a.index) - set(b.index))), 'only_without': ','.join(sorted(set(b.index) - set(a.index))),
                 'spearman_rf_common': spearmanr(a.loc[common, 'rf_importance_pct'], b.loc[common, 'rf_importance_pct'])[0] if len(common) > 2 else np.nan,
                 'spearman_shap_common': spearmanr(a.loc[common, 'shap_pct'], b.loc[common, 'shap_pct'])[0] if len(common) > 2 else np.nan,
                 'top5_with': ','.join(a.sort_values('shap_pct', ascending=False).index[:5]), 'top5_without': ','.join(b.sort_values('shap_pct', ascending=False).index[:5])})
S = pd.DataFrame(summ); S.to_csv(f'{OUT}/R3_32_summary.csv', index=False); print(S.T.to_string())
for kind in TARGETS:
    p = L[L.model == kind].pivot(index='feature', columns='setting', values=['rf_importance_pct', 'shap_pct']); p.columns = ['_'.join(c) for c in p.columns]
    p['delta_shap_pct'] = p['shap_pct_without_pollutants'] - p['shap_pct_with_pollutants']
    p.sort_values('shap_pct_without_pollutants', ascending=False).to_csv(f'{OUT}/R3_32_{kind}_side_by_side.csv'); print(kind); print(p.round(2).sort_values('shap_pct_without_pollutants', ascending=False).to_string())
