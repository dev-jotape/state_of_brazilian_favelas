import numpy as np, pandas as pd, geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, r2_score, f1_score
from common import *

X_VARS = ['age_index', 'median_age', 'sex_ratio', 'literate', 'race_wht', 'race_blk', 'race_ylw', 'race_brn', 'race_ind', 'race_unk',
          'depriv_idx', 'wtr_area', 'veg_area', 'risk_area', 'slope_10', 'slope_20', 'slope_30', 'slope_30p', 'hand_6p', 'hand_3_6',
          'hand_0_3', 'SO2', 'CO', 'O3', 'NO2', 'waste_coll', 'est_dens', 'est_edu', 'est_health', 'est_relig', 'est_agri', 'est_other',
          'est_const', 'intra_veg', 'prim_hosp', 'inp_hosp']
POLLUTANTS = ['SO2', 'CO', 'O3', 'NO2']
TARGETS = {'water': 'wtr_idx', 'sanitation': 'sanit_idx', 'regime': 'cluster_3'}


def load_pred():
    """Mirrors predict.ipynb load_data (dropna on all columns) and adds region/municipality ids from final_version."""
    gdf = gpd.read_file(CLUSTERS).dropna()
    fin = gpd.read_file(FINAL, ignore_geometry=True)[['CD_FCU', 'CD_MUN', 'CD_UF', 'SIGLA_UF']]
    df = pd.DataFrame(gdf.drop(columns='geometry')).merge(fin, on='CD_FCU', how='left')
    df['region'] = regions(df['CD_FCU'])
    return df


def split(df, feats, target, test_size=0.2, seed=42):
    """Same as predict.ipynb scale_and_split: scaler fit on ALL data, stratified (classification) split."""
    X = df[feats]; y = df[target]
    Xs = StandardScaler().fit_transform(X)
    strat = y if target == 'cluster_3' else None
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, stratify=strat, test_size=test_size, random_state=seed)
    Xs = pd.DataFrame(Xs, columns=feats)
    return Xs.iloc[tr].reset_index(drop=True), Xs.iloc[te].reset_index(drop=True), y.iloc[tr].values, y.iloc[te].values, tr, te


def make_rf(kind, seed=42):
    return (RandomForestClassifier if kind == 'regime' else RandomForestRegressor)(n_estimators=200, random_state=seed)


def run_rfecv(kind, Xtr, ytr, cv_splits=5, n_jobs=-1):
    cv = StratifiedKFold(cv_splits) if kind == 'regime' else KFold(cv_splits)
    scoring = 'accuracy' if kind == 'regime' else 'r2'
    r = RFECV(estimator=make_rf(kind), step=1, cv=cv, scoring=scoring, n_jobs=n_jobs)
    r.fit(Xtr, ytr)
    feats = list(np.array(Xtr.columns)[r.support_])
    return r, feats


def metric(kind, y, p):
    if kind == 'regime':
        return {'accuracy': accuracy_score(y, p), 'macro_f1': f1_score(y, p, average='macro')}
    return {'r2': r2_score(y, p)}
