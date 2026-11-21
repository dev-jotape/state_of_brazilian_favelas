"""Recipe taken from the original notebooks (not reconstructed):
 - water / sanitation: src/analysis/multi_corr.ipynb  -> clusters.shp, dropna, 36 X_vars, StandardScaler on all rows,
   RandomForestRegressor(n_estimators=50, random_state=42), RFECV(step=1, cv=KFold(5, shuffle=True, 42), scoring='r2'),
   fit on all complete rows (no hold-out).
 - regime: src/analysis/predict.ipynb -> stratified 80/20 split (random_state=42), RandomForestClassifier(n_estimators=200,
   random_state=42), RFECV(step=1, cv=StratifiedKFold(5), scoring='accuracy') on the 80% part.
RFECV n_jobs is limited (N_JOBS) to avoid freezing the machine; this does not change results."""
import numpy as np, pandas as pd, geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, r2_score
from common import *

N_JOBS = 4
X_VARS = ['age_index', 'median_age', 'sex_ratio', 'literate', 'race_wht', 'race_blk', 'race_ylw', 'race_brn', 'race_ind', 'race_unk',
          'depriv_idx', 'wtr_area', 'veg_area', 'risk_area', 'slope_10', 'slope_20', 'slope_30', 'slope_30p', 'hand_6p', 'hand_3_6',
          'hand_0_3', 'SO2', 'CO', 'O3', 'NO2', 'waste_coll', 'est_dens', 'est_edu', 'est_health', 'est_relig', 'est_agri', 'est_other',
          'est_const', 'intra_veg', 'prim_hosp', 'inp_hosp']
POLLUTANTS = ['SO2', 'CO', 'O3', 'NO2']
TARGETS = {'water': 'wtr_idx', 'sanitation': 'sanit_idx', 'regime': 'cluster_3'}


def load_pred():
    gdf = gpd.read_file(CLUSTERS).dropna()
    fin = gpd.read_file(FINAL, ignore_geometry=True)[['CD_FCU', 'CD_MUN', 'CD_UF']]
    df = pd.DataFrame(gdf.drop(columns='geometry')).merge(fin, on='CD_FCU', how='left')
    assert len(df) == len(gdf) and df['CD_MUN'].notna().all()
    df['region'] = regions(df['CD_FCU'])
    return df


def scaled(df, feats):
    return pd.DataFrame(StandardScaler().fit_transform(df[feats]), columns=feats)


def make_rf(kind):
    return RandomForestClassifier(n_estimators=200, random_state=42) if kind == 'regime' else RandomForestRegressor(n_estimators=50, random_state=42)


def select_features(df, kind, feats_all):
    """returns (selected feature list, RF importance dict of the RFECV final estimator, n_rows_used)"""
    x = scaled(df, feats_all); y = df[TARGETS[kind]].values
    if kind == 'regime':
        idx = np.arange(len(df))
        tr, te = train_test_split(idx, stratify=y, test_size=0.2, random_state=42)
        x, y = x.iloc[tr].reset_index(drop=True), y[tr]
        cv, sc = StratifiedKFold(5), 'accuracy'
    else:
        cv, sc = KFold(5, shuffle=True, random_state=42), 'r2'
    r = RFECV(estimator=make_rf(kind), step=1, cv=cv, scoring=sc, n_jobs=N_JOBS).fit(x, y)
    feats = list(np.array(feats_all)[r.support_])
    return feats, dict(zip(feats, map(float, r.estimator_.feature_importances_))), len(y), float(np.max(r.cv_results_['mean_test_score']))
