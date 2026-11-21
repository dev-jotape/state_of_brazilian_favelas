"""R3.34 - writes MODEL_DESCRIPTION.md describing what the original notebooks actually do (multi_corr.ipynb, predict.ipynb)."""
import json, platform, importlib.metadata as md
from pred_common import *
sel = json.load(open(f'{OUT}/01_selection.json')); ins = json.load(open(f'{OUT}/insample_r2_multi_corr_recipe.json'))
v = {p: md.version(p) for p in ['scikit-learn', 'xgboost', 'shap', 'numpy', 'pandas', 'scipy', 'geopandas', 'joblib']}
names = {'water': 'Water index (`wtr_idx`)', 'sanitation': 'Sanitation index (`sanit_idx`)', 'regime': 'Regime (`cluster_3`)'}
L = []; A = L.append
A('# Predictive models: description of what the code does (R3.34)\n')
A('Sources: `src/analysis/multi_corr.ipynb` (water, sanitation) and `src/analysis/predict.ipynb` (regime). Selections re-run in `src/robustness/01_select_features.py` with the same settings.\n')
A('## Data\n')
A('- File `dataset/shp/clusters.shp` (the notebooks read this path; the file shipped in this checkout is `dataset/clusters/clusters.shp`, 12,344 rows; there is no `dataset/shp/clusters.shp` in this checkout, so that path in the notebooks does not resolve here), `dropna()` over all columns → 11,503 favelas.')
A(f'- {len(X_VARS)} candidate predictors (same list in both notebooks): ' + ', '.join(f'`{x}`' for x in X_VARS) + '.')
A('- Predictors are standardised with `StandardScaler` fitted on all 11,503 rows. No log transform (commented out), no imputation.\n')
A('## Settings as implemented\n')
A('| | Water / sanitation (`multi_corr.ipynb`) | Regime (`predict.ipynb`) |\n|---|---|---|')
A('| Estimator | `RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)`; other arguments default (`max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=1.0, criterion="squared_error", bootstrap=True`) | `RandomForestClassifier(n_estimators=200, random_state=42)`; other arguments default (`max_features="sqrt", criterion="gini"`) |')
A('| Hyper-parameter search | none | none |')
A('| Rows used | all 11,503 (no hold-out split in the notebook) | stratified `train_test_split(test_size=0.2, random_state=42)`; RFECV on the 9,202 training rows, SHAP on the 2,301 held-out rows |')
A('| Feature selection | `RFECV(step=1, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")` | `RFECV(step=1, cv=StratifiedKFold(5), scoring="accuracy")` (no shuffle) |')
A('| Importance reported | impurity importance of `rfecv.estimator_` (refit on the selected features); `rf_model.fit(x[features], y)` is refit on all rows for SHAP | impurity importance of the RF refit on the selected features; `shap.TreeExplainer`, test rows |')
A('| SHAP | `shap.TreeExplainer(rf_model)` on all rows, mean(\\|SHAP\\|) as % | `shap.TreeExplainer`, test rows, mean(\\|SHAP\\|) summed over classes |')
A('| Metric printed in notebook | `r2_score` on the same rows used for fitting (in-sample) | accuracy / classification report on the hold-out |')
A('| Seed | `random_state=42` (RF, KFold, split) | `random_state=42` (RF, split) |\n')
A('The `predict.ipynb` notebook also contains `LogisticRegressionCV` (L1, liblinear, `Cs=np.logspace(-4,4,20)`, `StratifiedKFold(5)`, accuracy) and an RF with `SequentialFeatureSelector(forward, n_features_to_select="auto", StratifiedKFold(5))`; these are not used in the robustness runs.\n')
A('## Final predictors selected by RFECV (re-run, with pollutants)\n')
for k in ('water', 'sanitation', 'regime'):
    s = sel[f'with_pollutants/{k}']
    A(f'**{names[k]}**, {len(s["selected"])} of {s["n_candidates"]} (RFECV best mean CV score {s["rfecv_best_cv_score"]:.4f}): ' + ', '.join(f'`{f}` ({100*i:.1f}%)' for f, i in sorted(s['rf_importance'].items(), key=lambda x: -x[1])) + '\n')
A('Percentages: impurity importance of the final estimator.\n')
A('## Selected predictors without the Sentinel-5P pollutants (SO2, CO, O3, NO2)\n')
for k in ('water', 'sanitation', 'regime'):
    s = sel[f'without_pollutants/{k}']
    A(f'**{names[k]}**, {len(s["selected"])} of {s["n_candidates"]} (RFECV best mean CV score {s["rfecv_best_cv_score"]:.4f}): ' + ', '.join(f'`{f}` ({100*i:.1f}%)' for f, i in sorted(s['rf_importance'].items(), key=lambda x: -x[1])) + '\n')
A('## Metrics produced by this recipe (for reference)\n')
A('| Model | In-sample R² (as printed by the notebook) | RFECV best mean CV score (inner CV) |\n|---|---|---|')
for k in ('water', 'sanitation'):
    A(f'| {names[k]} | {ins[k]:.4f} | {sel[f"with_pollutants/{k}"]["rfecv_best_cv_score"]:.4f} (R², KFold 5) |')
A(f'| {names["regime"]} | n/a | {sel["with_pollutants/regime"]["rfecv_best_cv_score"]:.4f} (accuracy, StratifiedKFold 5, training part) |')
A('\n## Software\n')
A(f'- Python {platform.python_version()} (README: 3.13.12); ' + ', '.join(f'{p} {x}' for p, x in v.items()) + ' (pinned in `requirements.txt`).')
A('- CPU only. In the robustness runs `RFECV(n_jobs=4)`; RF estimators use their default single-thread fit (the notebooks use `n_jobs=-1`); this does not change results.\n')
A('## Divergences between README and code\n')
A('- README says 10-fold CV; the code uses 5-fold (`KFold(5)` / `StratifiedKFold(5)`).')
A('- The README R² (≈0.91 water, ≈0.92 sanitation) matches the **in-sample** R² of the multi_corr recipe (table above), not a cross-validated or hold-out R².')
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MODEL_DESCRIPTION.md'), 'w').write('\n'.join(L) + '\n')
