"""Shared helpers for the robustness analyses (Lote 4). Does NOT modify existing results.

Re-implements, with identical logic, the index construction of
src/cluster/generate_clusters.ipynb, parametrised by specification variants.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'dataset', 'robustness')
os.makedirs(OUT, exist_ok=True)
SLUM = os.path.join(ROOT, 'dataset/favelas_urban_communities/slum_variables/final/slum_variables.shp')
CLUSTERS = os.path.join(ROOT, 'dataset/clusters/clusters.shp')
FINAL = os.path.join(ROOT, 'dataset/shp/final_version.shp')

IDX = ['wtr_idx', 'sanit_idx', 'living_area_idx', 'durable_house_idx']
REGION = {'1': 'North', '2': 'Northeast', '3': 'Southeast', '4': 'South', '5': 'Center-West'}

PCT_COLS = [
    'wtr_pipe', 'wtr_semi', 'no_wtr_pip', 'wtr_dist', 'wtr_deep', 'wtr_shal', 'wtr_spring', 'wtr_truck', 'wtr_rain', 'wtr_river', 'wtr_oth',
    'sanit_1', 'sanit_2', 'sanit_3', 'sanit_4', 'sanit_5', 'sanit_6', 'sanit_7', 'no_bath', 'excl_bath', 'shrd_bath', 'toilet',
    'hh_type_1', 'hh_type_2', 'hh_type_3', 'hh_type_4', 'hh_type_5', 'hh_type_6']


def load_slum():
    gdf = gpd.read_file(SLUM, ignore_geometry=True)
    gdf = gdf[~gdf['wtr_pipe'].isna()].copy()
    gdf[PCT_COLS] = gdf[PCT_COLS] / 100
    return gdf.reset_index(drop=True)


def _w(spec, n):
    """weights for an ordinal component with n levels (worst -> best)."""
    if spec == 'exp':
        return [2 ** i for i in range(n)]
    if spec == 'linear':
        return [i + 1 for i in range(n)]
    raise ValueError(spec)


def _comp(levels, weights):
    return sum(l * w for l, w in zip(levels, weights))


def living_area_raw(g):
    """Living-area index exactly as in the notebook (z-scores, quartile categories, min-max, inverted)."""
    z_hh = (g['hh_density'] - g['hh_density'].mean()) / g['hh_density'].std()
    z_pop = (g['demo_dens'] - g['demo_dens'].mean()) / g['demo_dens'].std()
    hh_cat = np.select([z_hh < z_hh.median(), z_hh > z_hh.quantile(.75)], [1, 3], default=2)
    pop_cat = np.select([z_pop < z_pop.median(), z_pop > z_pop.quantile(.75)], [1, 3], default=2)
    v = (hh_cat * z_hh) + (pop_cat * z_pop)
    return 1 - MinMaxScaler().fit_transform(v.values.reshape(-1, 1)).ravel()


def build_indices(g, weights='exp', norm='data'):
    """weights: 'exp' (paper), 'linear' (1,2,3...), 'equal' (best-category share only, no gradation).
    norm: 'data' (min-max on the observed data, paper) or 'theory' (min-max on the theoretical interval
    of the raw score; living-area keeps data min-max because it has no bounded theoretical range)."""
    if weights in ('exp', 'linear'):
        w2, w3 = _w(weights, 2), _w(weights, 3)
        wtr_infra = _comp([g.wtr_semi + g.no_wtr_pip, g.wtr_pipe], w2)
        wtr_src = _comp([g.wtr_deep + g.wtr_shal + g.wtr_spring + g.wtr_truck + g.wtr_rain + g.wtr_river + g.wtr_oth, g.wtr_dist], w2)
        s_type = _comp([g.sanit_5 + g.sanit_6 + g.sanit_7 + g.no_bath, g.sanit_2 + g.sanit_3 + g.sanit_4, g.sanit_1], w3)
        s_bath = _comp([g.shrd_bath + g.toilet, g.excl_bath], w2)
        dur = _comp([g.hh_type_4 + g.hh_type_5 + g.hh_type_6, g.hh_type_1 + g.hh_type_2 + g.hh_type_3], w2)
        wtr = (wtr_infra + wtr_src) / 2
        san = (s_type + s_bath) / 2
        # theoretical [min, max] of each raw score (shares sum to 1)
        th = {'wtr': (min(w2), max(w2)), 'san': ((min(w3) + min(w2)) / 2, (max(w3) + max(w2)) / 2), 'dur': (min(w2), max(w2))}
    elif weights == 'equal':
        wtr = (g.wtr_pipe + g.wtr_dist) / 2
        san = (g.sanit_1 + g.excl_bath) / 2
        dur = g.hh_type_1 + g.hh_type_2 + g.hh_type_3
        th = {'wtr': (0, 1), 'san': (0, 1), 'dur': (0, 1)}
    else:
        raise ValueError(weights)
    out = pd.DataFrame(index=g.index)
    for name, raw, key in [('wtr_idx', wtr, 'wtr'), ('sanit_idx', san, 'san')]:
        out[name] = _norm(raw, th[key], norm)
    out['living_area_idx'] = living_area_raw(g)
    out['durable_house_idx'] = _norm(dur, th['dur'], norm)
    return out[IDX]


def _norm(raw, th, norm):
    raw = np.asarray(raw, dtype=float)
    if norm == 'data':
        return (raw - raw.min()) / (raw.max() - raw.min())
    return (raw - th[0]) / (th[1] - th[0])


def kmeans_labels(X, k=3, seed=42):
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(np.asarray(X))


def paper_labels(raw_labels):
    """Same relabeling as the notebook: swap 0 <-> 1."""
    return pd.Series(raw_labels).replace({1: 0, 0: 1, 2: 2}).values


def regions(g_cd_fcu):
    return pd.Series(g_cd_fcu).astype(str).str[0].map(REGION).values
