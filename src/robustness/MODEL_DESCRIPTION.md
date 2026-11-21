# Predictive models: description of what the code does (R3.34)

Sources: `src/analysis/multi_corr.ipynb` (water, sanitation) and `src/analysis/predict.ipynb` (regime). Selections re-run in `src/robustness/01_select_features.py` with the same settings.

## Data

- File `dataset/shp/clusters.shp` (the notebooks read this path; the file shipped in this checkout is `dataset/clusters/clusters.shp`, 12,344 rows; there is no `dataset/shp/clusters.shp` in this checkout, so that path in the notebooks does not resolve here), `dropna()` over all columns → 11,503 favelas.
- 36 candidate predictors (same list in both notebooks): `age_index`, `median_age`, `sex_ratio`, `literate`, `race_wht`, `race_blk`, `race_ylw`, `race_brn`, `race_ind`, `race_unk`, `depriv_idx`, `wtr_area`, `veg_area`, `risk_area`, `slope_10`, `slope_20`, `slope_30`, `slope_30p`, `hand_6p`, `hand_3_6`, `hand_0_3`, `SO2`, `CO`, `O3`, `NO2`, `waste_coll`, `est_dens`, `est_edu`, `est_health`, `est_relig`, `est_agri`, `est_other`, `est_const`, `intra_veg`, `prim_hosp`, `inp_hosp`.
- Predictors are standardised with `StandardScaler` fitted on all 11,503 rows. No log transform (commented out), no imputation.

## Settings as implemented

| | Water / sanitation (`multi_corr.ipynb`) | Regime (`predict.ipynb`) |
|---|---|---|
| Estimator | `RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)`; other arguments default (`max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=1.0, criterion="squared_error", bootstrap=True`) | `RandomForestClassifier(n_estimators=200, random_state=42)`; other arguments default (`max_features="sqrt", criterion="gini"`) |
| Hyper-parameter search | none | none |
| Rows used | all 11,503 (no hold-out split in the notebook) | stratified `train_test_split(test_size=0.2, random_state=42)`; RFECV on the 9,202 training rows, SHAP on the 2,301 held-out rows |
| Feature selection | `RFECV(step=1, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")` | `RFECV(step=1, cv=StratifiedKFold(5), scoring="accuracy")` (no shuffle) |
| Importance reported | impurity importance of `rfecv.estimator_` (refit on the selected features); `rf_model.fit(x[features], y)` is refit on all rows for SHAP | impurity importance of the RF refit on the selected features; `shap.TreeExplainer`, test rows |
| SHAP | `shap.TreeExplainer(rf_model)` on all rows, mean(\|SHAP\|) as % | `shap.TreeExplainer`, test rows, mean(\|SHAP\|) summed over classes |
| Metric printed in notebook | `r2_score` on the same rows used for fitting (in-sample) | accuracy / classification report on the hold-out |
| Seed | `random_state=42` (RF, KFold, split) | `random_state=42` (RF, split) |

The `predict.ipynb` notebook also contains `LogisticRegressionCV` (L1, liblinear, `Cs=np.logspace(-4,4,20)`, `StratifiedKFold(5)`, accuracy) and an RF with `SequentialFeatureSelector(forward, n_features_to_select="auto", StratifiedKFold(5))`; these are not used in the robustness runs.

## Final predictors selected by RFECV (re-run, with pollutants)

**Water index (`wtr_idx`)**, 8 of 36 (RFECV best mean CV score 0.4176): `CO` (25.7%), `waste_coll` (12.5%), `O3` (12.3%), `prim_hosp` (10.5%), `age_index` (10.3%), `NO2` (10.2%), `depriv_idx` (10.2%), `est_dens` (8.3%)

**Sanitation index (`sanit_idx`)**, 20 of 36 (RFECV best mean CV score 0.4939): `depriv_idx` (25.1%), `age_index` (7.9%), `est_dens` (6.6%), `NO2` (6.3%), `CO` (5.4%), `waste_coll` (4.9%), `O3` (4.5%), `hand_6p` (4.3%), `inp_hosp` (3.9%), `slope_10` (3.7%), `prim_hosp` (3.6%), `literate` (3.5%), `sex_ratio` (3.0%), `SO2` (2.7%), `race_blk` (2.7%), `race_wht` (2.6%), `hand_0_3` (2.6%), `veg_area` (2.2%), `hand_3_6` (2.2%), `race_brn` (2.2%)

**Regime (`cluster_3`)**, 15 of 36 (RFECV best mean CV score 0.7136): `depriv_idx` (11.0%), `NO2` (9.4%), `age_index` (7.7%), `CO` (7.5%), `O3` (7.4%), `est_dens` (7.0%), `inp_hosp` (6.1%), `slope_10` (5.9%), `waste_coll` (5.6%), `race_brn` (5.6%), `prim_hosp` (5.5%), `literate` (5.5%), `SO2` (5.4%), `hand_0_3` (5.4%), `race_wht` (5.0%)

Percentages: impurity importance of the final estimator.

## Selected predictors without the Sentinel-5P pollutants (SO2, CO, O3, NO2)

**Water index (`wtr_idx`)**, 29 of 32 (RFECV best mean CV score 0.2917): `waste_coll` (13.3%), `race_brn` (9.3%), `age_index` (7.4%), `prim_hosp` (7.1%), `depriv_idx` (6.7%), `literate` (5.5%), `est_dens` (4.8%), `inp_hosp` (4.4%), `race_blk` (3.8%), `sex_ratio` (3.7%), `hand_3_6` (3.5%), `race_wht` (3.3%), `intra_veg` (3.2%), `veg_area` (3.1%), `hand_0_3` (2.9%), `slope_10` (2.5%), `median_age` (2.5%), `slope_20` (2.4%), `hand_6p` (2.1%), `race_ylw` (2.0%), `race_ind` (1.8%), `slope_30` (1.3%), `risk_area` (1.0%), `slope_30p` (0.7%), `est_relig` (0.4%), `est_edu` (0.4%), `est_health` (0.3%), `est_agri` (0.2%), `est_other` (0.1%)

**Sanitation index (`sanit_idx`)**, 26 of 32 (RFECV best mean CV score 0.4527): `depriv_idx` (26.1%), `est_dens` (7.7%), `age_index` (7.4%), `waste_coll` (5.7%), `hand_6p` (4.6%), `inp_hosp` (4.1%), `race_brn` (3.7%), `race_blk` (3.6%), `literate` (3.6%), `prim_hosp` (3.6%), `slope_10` (3.4%), `race_wht` (3.3%), `sex_ratio` (3.1%), `hand_0_3` (2.7%), `hand_3_6` (2.5%), `veg_area` (2.2%), `median_age` (2.2%), `slope_20` (2.0%), `race_ind` (1.5%), `slope_30` (1.5%), `intra_veg` (1.5%), `risk_area` (1.3%), `race_ylw` (1.2%), `slope_30p` (0.9%), `est_relig` (0.4%), `est_const` (0.3%)

**Regime (`cluster_3`)**, 27 of 32 (RFECV best mean CV score 0.6883): `depriv_idx` (11.8%), `age_index` (6.1%), `race_brn` (6.0%), `est_dens` (5.9%), `waste_coll` (5.5%), `literate` (5.3%), `inp_hosp` (4.8%), `slope_10` (4.7%), `prim_hosp` (4.4%), `race_wht` (4.2%), `race_blk` (4.1%), `sex_ratio` (3.7%), `hand_3_6` (3.6%), `hand_6p` (3.6%), `hand_0_3` (3.6%), `median_age` (3.4%), `slope_20` (3.3%), `veg_area` (3.1%), `intra_veg` (2.6%), `slope_30` (2.3%), `race_ind` (2.0%), `race_ylw` (1.7%), `risk_area` (1.5%), `slope_30p` (1.4%), `est_relig` (0.6%), `est_edu` (0.5%), `est_health` (0.3%)

## Metrics produced by this recipe (for reference)

| Model | In-sample R² (as printed by the notebook) | RFECV best mean CV score (inner CV) |
|---|---|---|
| Water index (`wtr_idx`) | 0.9157 | 0.4176 (R², KFold 5) |
| Sanitation index (`sanit_idx`) | 0.9273 | 0.4939 (R², KFold 5) |
| Regime (`cluster_3`) | n/a | 0.7136 (accuracy, StratifiedKFold 5, training part) |

## Software

- Python 3.13.13 (README: 3.13.12); scikit-learn 1.6.1, xgboost 3.0.2, shap 0.48.0, numpy 2.2.4, pandas 2.2.3, scipy 1.15.2, geopandas 1.0.1, joblib 1.4.2 (pinned in `requirements.txt`).
- CPU only. In the robustness runs `RFECV(n_jobs=4)`; RF estimators use their default single-thread fit (the notebooks use `n_jobs=-1`); this does not change results.

## Divergences between README and code

- README says 10-fold CV; the code uses 5-fold (`KFold(5)` / `StratifiedKFold(5)`).
- The README R² (≈0.91 water, ≈0.92 sanitation) matches the **in-sample** R² of the multi_corr recipe (table above), not a cross-validated or hold-out R².
