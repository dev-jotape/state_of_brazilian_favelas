"""Recreate Fig. 3 (distribution of indicators by regime, four sub-figures a-d).

Same variables, panel sizes and layout as src/analysis/boxplot.ipynb (cell 23),
but with the shared regime palette (palette.py), regime names and an in-figure
legend (R1.15/R3.25) on panel (a). Vector PDF output (E.10).

Usage (from the repository root):
    python src/figures/fig3_boxplots.py \
        --data dataset/shp/final_version_no_missing_values.shp --outdir figures
writes boxplot_demographic.pdf, boxplot_topography.pdf, boxplot_services.pdf,
boxplot_polution.pdf (file names as used in sn-article.tex).
"""
import argparse
import os
import sys

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(__file__))
from palette import REGIME_COLORS, REGIME_EDGE, REGIME_NAMES  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

PRETTY = {
    "age_idx": "Age Index", "literate": "Literacy\n(%)", "depriv_idx": "Deprivation Index",
    "race_wht": "Race - White\n(%)", "race_brn": "Race - Brown/Pardo\n(%)",
    "race_blk": "Race - Black\n(%)", "CO": "CO", "O3": r"$O_3$", "SO2": r"$SO_2$",
    "NO2": r"$NO_2$", "slope_10": "Slope 0-10%", "slope_20": "Slope 10-20%",
    "slope_30": "Slope 20-30%", "slope_30p": "Slope >30%", "hand_0_3": "HAND 0-3m",
    "hand_3_6": "HAND 3-6m", "hand_6p": "HAND >6m",
    "log_prim_hosp": "Primary Hospitals\n(Distance - log)",
    "log_inp_hosp": "Inpatient Hospitals\n(Distance - log)",
    "waste_coll": "Waste Collection\n(%)", "log_est_dens": "Commercial and service\nestablishment density (log)",
    "log_population": "Population\n(log)",
}
# (file name, variables); legend is drawn on the first one only
FIGS = [
    ("boxplot_demographic", ["log_population", "age_idx", "literate", "depriv_idx",
                             "race_wht", "race_brn", "race_blk"]),
    ("boxplot_topography", ["slope_10", "slope_20", "slope_30", "slope_30p",
                            "hand_0_3", "hand_3_6", "hand_6p"]),
    ("boxplot_services", ["log_prim_hosp", "log_inp_hosp", "waste_coll", "log_est_dens"]),
    ("boxplot_polution", ["CO", "O3", "SO2", "NO2"]),
]
BOX_W, BOX_H, H_GAP = 1.1, 3.4, 0.75
EXTRA_RIGHT = {"boxplot_services": 0.7}  # inches of white space so the long last x label is not clipped
SCI_TICKS = {"boxplot_polution"}  # tiny values (1e-4): scientific notation keeps panel width


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/shp/final_version_no_missing_values.shp")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--ext", default="pdf", help="pdf (journal, vector) or png (preview)")
    a = ap.parse_args()

    gdf = gpd.read_file(a.data, ignore_geometry=True)
    for col, src in [("log_population", "population"), ("log_prim_hosp", "prim_hosp"),
                     ("log_inp_hosp", "inp_hosp"), ("log_est_dens", "est_dens")]:
        gdf[col] = np.log(gdf[src].replace(0, np.nan))
    gdf["cluster"] = gdf["cluster"].astype(int)
    order = sorted(REGIME_COLORS)
    pal = [REGIME_COLORS[k] for k in order]
    sns.set_theme(style="whitegrid")

    for fi, (name, vars_) in enumerate(FIGS):
        n = len(vars_)
        gap = H_GAP
        fig, axes = plt.subplots(1, n, figsize=(n * BOX_W + (n - 1) * gap + EXTRA_RIGHT.get(name, 0), BOX_H))
        axes = np.atleast_1d(axes)
        for ax, var in zip(axes, vars_):
            d = gdf[[var, "cluster"]].dropna()
            sns.boxplot(data=d, x=[0] * len(d), y=var, hue="cluster", hue_order=order,
                        palette=pal, ax=ax, linewidth=0.7, fliersize=1.2, width=0.01,
                        legend=False)
            for patch in ax.patches:  # thin outline keeps the lightest blue visible
                patch.set_edgecolor(REGIME_EDGE)
            ax.set_title(""); ax.set_ylabel("")
            ax.set_xlabel(PRETTY[var], fontsize=10, labelpad=4)
            ax.set_xticklabels([]); ax.tick_params(axis="x", length=0)
            ax.tick_params(axis="y", labelsize=10, pad=1)
            if name in SCI_TICKS:
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
                ax.yaxis.get_offset_text().set_fontsize(9)
            ax.set_facecolor("white"); ax.grid(True, color="#ececec", linewidth=0.85)
            ax.margins(x=0.10)
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color("#888888"); sp.set_linewidth(1.55)
        top = 0.9 if name in SCI_TICKS else 0.98
        if fi == 0:  # legend of the regime colours (R1.15 / R3.25)
            handles = [Patch(facecolor=REGIME_COLORS[k], edgecolor=REGIME_EDGE,
                             label=REGIME_NAMES[k]) for k in order]
            fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
                       fontsize=10, bbox_to_anchor=(0.5, 1.0), title="Regime",
                       title_fontsize=10, alignment="center")
            top = 0.84
        fig.subplots_adjust(wspace=gap / BOX_W, top=top, left=0.05 * (n * BOX_W + (n - 1) * gap) / fig.get_figwidth(), bottom=0.27,
                            right=1 - (0.015 * (n * BOX_W + (n - 1) * gap) + EXTRA_RIGHT.get(name, 0)) / fig.get_figwidth())
        out = os.path.join(a.outdir, f"{name}.{a.ext}")
        fig.savefig(out)
        print("saved", out)


if __name__ == "__main__":
    main()
