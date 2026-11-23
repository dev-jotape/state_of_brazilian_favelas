"""Recreate Figs. 5 and 6 (SHAP of the Random Forest models for the water and sanitation indices).

Recipe of src/analysis/multi_corr.ipynb: clusters.shp (rows with no missing values), StandardScaler on the 36
candidate indicators, RandomForestRegressor(n_estimators=50, random_state=42), TreeExplainer. To avoid re-running the
slow RFECV, the selected indicators are read from the saved RFECV result of the robustness scripts
(dataset/robustness/01_selection.json, keys 'with_pollutants/water' and 'with_pollutants/sanitation').
Left: mean(|SHAP|) as % of the total (neutral grey bars, R1.18). Right: SHAP value of each favela, coloured by the
indicator value with a colour-blind-safe sequential scale (viridis) instead of red-green (R1.8). Larger y-axis fonts (R1.19).

Usage (from the repository root):
    python src/figures/fig56_shap.py --outdir figures
writes shap_water.pdf and shap_sanitation.pdf (use --ext png for a preview).
"""
import argparse
import json
import os

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

X_VARS = ['age_index', 'median_age', 'sex_ratio', 'literate', 'race_wht', 'race_blk', 'race_ylw', 'race_brn',
          'race_ind', 'race_unk', 'depriv_idx', 'wtr_area', 'veg_area', 'risk_area', 'slope_10', 'slope_20',
          'slope_30', 'slope_30p', 'hand_6p', 'hand_3_6', 'hand_0_3', 'SO2', 'CO', 'O3', 'NO2', 'waste_coll',
          'est_dens', 'est_edu', 'est_health', 'est_relig', 'est_agri', 'est_other', 'est_const', 'intra_veg',
          'prim_hosp', 'inp_hosp']
NAMES = {
    'age_index': 'Age Index', 'median_age': 'Median Age', 'sex_ratio': 'Sex Ratio', 'literate': 'Literate',
    'race_wht': 'White', 'race_blk': 'Black', 'race_brn': 'Brown', 'race_unk': 'Unknown Race',
    'depriv_idx': 'Deprivation Index', 'wtr_area': 'Water Area', 'veg_area': 'Vegetation Cover',
    'risk_area': 'Risk Area', 'slope_10': 'Slope >10%', 'hand_0_3': 'HAND 0–3 m', 'hand_3_6': 'HAND 3–6 m',
    'hand_6p': 'HAND >6 m', 'SO2': 'Sulfur Dioxide (SO₂)', 'CO': 'Carbon Monoxide (CO)', 'O3': 'Ozone (O₃)',
    'NO2': 'Nitrogen Dioxide (NO₂)', 'waste_coll': 'Waste Collection',
    'est_dens': 'Commercial and service\nestablishment density', 'prim_hosp': 'Primary Hospital',
    'inp_hosp': 'Inpatient Hospital',
}
BAR_COLOR = "#7F7F7F"   # neutral grey: not a regime colour (R1.18)
DOT_CMAP = "viridis"    # colour-blind-safe sequential scale (replaces RdYlGn)
FS = 13                 # y-axis tick font size (R1.19)


def plot(shap_values, X, feats, out):
    shap_mean = np.abs(shap_values).mean(axis=0)
    pct = pd.Series(100 * shap_mean / shap_mean.sum(), index=feats).sort_values(ascending=False)
    order = pct.index.tolist()
    idx = [feats.index(f) for f in order]
    labels = [NAMES.get(f, f) for f in order]
    n = len(order)
    fig, axes = plt.subplots(1, 2, figsize=(17, 0.5 * n + 2.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.barh(np.arange(n), pct.values, color=BAR_COLOR)
    ax.set_yticks(np.arange(n)); ax.set_yticklabels(labels, fontsize=FS)
    ax.invert_yaxis()
    ax.set_xlabel("mean(|SHAP value|) (%)", fontsize=FS); ax.tick_params(axis="x", labelsize=FS - 1)
    ax.set_title("Global SHAP (bar)", fontsize=FS + 1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.sca(axes[1])
    shap.summary_plot(shap_values[:, idx], X[order], plot_type="dot", sort=False, show=False,
                      max_display=n, cmap=DOT_CMAP, feature_names=labels, plot_size=None)
    a2 = axes[1]
    a2.set_title("SHAP – value per favela (dot)", fontsize=FS + 1)
    a2.set_xlabel("SHAP value (impact on model output)", fontsize=FS)
    a2.set_yticklabels([]); a2.tick_params(axis="y", left=False)  # rows align with the bars on the left
    a2.tick_params(axis="x", labelsize=FS - 1)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", default="dataset/clusters/clusters.shp")
    ap.add_argument("--selection", default="dataset/robustness/01_selection.json")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--ext", default="pdf")
    ap.add_argument("--reuse", action="store_true", help="reuse <name>_values.npy from --outdir instead of recomputing SHAP")
    ap.add_argument("--only", default=None, help="water or sanitation (default: both)")
    a = ap.parse_args()

    gdf = gpd.read_file(a.clusters).dropna()
    x = pd.DataFrame(StandardScaler().fit_transform(gdf[X_VARS]), columns=X_VARS)
    sel = json.load(open(a.selection))
    for key, target, name in [("with_pollutants/water", "wtr_idx", "shap_water"),
                              ("with_pollutants/sanitation", "sanit_idx", "shap_sanitation")]:
        if a.only and a.only not in key:
            continue
        feats = sel[key]["selected"]
        npy = os.path.join(a.outdir, f"{name}_values.npy")
        if a.reuse and os.path.exists(npy):
            sv = np.load(npy)
        else:
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1).fit(x[feats], gdf[target])
            # TreeExplainer is single-threaded: explain row chunks in parallel (same values)
            expl = shap.TreeExplainer(rf)
            chunks = np.array_split(np.arange(len(x)), 8)
            sv = np.vstack(Parallel(n_jobs=4)(delayed(expl.shap_values)(x[feats].iloc[c], check_additivity=False)
                                              for c in chunks))
            pct = 100 * np.abs(sv).mean(0) / np.abs(sv).mean(0).sum()
            print(key, "in-sample R2 = %.4f" % rf.score(x[feats], gdf[target]))
            print(pd.Series(pct, index=feats).sort_values(ascending=False).round(2).to_string(), flush=True)
            np.save(os.path.join(a.outdir, f"{name}_values.npy"), sv)
        plot(sv, x[feats], feats, os.path.join(a.outdir, f"{name}.{a.ext}"))


if __name__ == "__main__":
    main()
