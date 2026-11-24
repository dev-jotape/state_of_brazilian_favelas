"""Recreate Fig. 8 (regional inequality): (a) state-level water/sanitation scatter, (b) share of favelas in each
regime by macro-region, (c) Sankey of macro-regions -> regimes. Same layout as in the article.

Colours: regimes use the shared Okabe-Ito palette (palette.py: blue/yellow/vermilion); macro-regions in (a) use
colours that avoid those three (green, pink, purple, dark grey, brown) and different marker shapes (R1.20).
The Sankey is drawn with matplotlib (vector, editable text) instead of plotly.

Usage (from the repository root):
    python src/figures/fig8_regional.py --data dataset/shp/final_version_no_missing_values.shp --out figures/regions_plot.pdf
"""
import argparse
import os
import sys

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path

sys.path.insert(0, os.path.dirname(__file__))
from palette import REGIME_COLORS, REGIME_EDGE, REGIME_NAMES  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

UF_REGION = {
    **{u: "North" for u in ["AC", "AP", "AM", "PA", "RO", "RR", "TO"]},
    **{u: "Northeast" for u in ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"]},
    **{u: "Center-West" for u in ["DF", "GO", "MT", "MS"]},
    **{u: "Southeast" for u in ["ES", "MG", "RJ", "SP"]},
    **{u: "South" for u in ["PR", "RS", "SC"]},
}
REGIONS = ["North", "Northeast", "Center-West", "Southeast", "South"]
# region colours: none of the three regime colours (blue, yellow, vermilion)
REGION_COLORS = {"North": "#009E73", "Northeast": "#CC79A7", "Center-West": "#6A3D9A",
                 "Southeast": "#666666", "South": "#8C510A"}
REGION_MARKERS = {"North": "o", "Northeast": "s", "Center-West": "^", "Southeast": "D", "South": "P"}
SANKEY_LEFT_ORDER = ["Southeast", "South", "Northeast", "Center-West", "North"]


def sankey(ax, counts):
    """counts: DataFrame regions x regimes (number of favelas)."""
    total = counts.values.sum()
    gap = 0.012 * total
    left_tot = counts.sum(axis=1)[SANKEY_LEFT_ORDER]
    right_tot = counts.sum(axis=0)
    H = total + gap * (len(left_tot) - 1)
    # node positions (y from top, drawn with y inverted)
    ly, y = {}, 0.0
    for r in SANKEY_LEFT_ORDER:
        ly[r] = y; y += left_tot[r] + gap
    ry, y = {}, 0.0
    scale_gap = (H - right_tot.sum()) / (len(right_tot) - 1)
    for k in sorted(right_tot.index):
        ry[k] = y; y += right_tot[k] + scale_gap
    xl, xr, w = 0.0, 1.0, 0.012
    lcur, rcur = dict(ly), dict(ry)
    for k in sorted(right_tot.index):          # flows drawn per regime so colours stack nicely
        for r in SANKEY_LEFT_ORDER:
            v = counts.loc[r, k]
            if v == 0:
                continue
            y0, y1 = lcur[r], rcur[k]
            lcur[r] += v; rcur[k] += v
            xm = (xl + w + xr) / 2
            verts = [(xl + w, y0), (xm, y0), (xm, y1), (xr, y1), (xr, y1 + v), (xm, y1 + v), (xm, y0 + v), (xl + w, y0 + v), (xl + w, y0)]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
            ax.add_patch(PathPatch(Path(verts, codes), facecolor=REGIME_COLORS[k], alpha=0.38, edgecolor="none"))
    for r in SANKEY_LEFT_ORDER:
        ax.add_patch(Rectangle((xl, ly[r]), w, left_tot[r], facecolor="#cccccc", edgecolor="#999999", lw=0.6))
        ax.text(xl + w + 0.006, ly[r] + left_tot[r] / 2, r, ha="left", va="center", fontsize=8, color="#222222")
    for k in sorted(right_tot.index):
        ax.add_patch(Rectangle((xr, ry[k]), w, right_tot[k], facecolor=REGIME_COLORS[k], edgecolor=REGIME_EDGE, lw=0.6))
        ax.text(xr - 0.006, ry[k] + right_tot[k] / 2, REGIME_NAMES[k], ha="right", va="center", fontsize=8, color="#222222")
    ax.set_xlim(0, 1 + w); ax.set_ylim(H, 0); ax.axis("off")


def panel_title(ax, letter, title, x_title=0.055, y=1.03, **kw):
    ax.text(0, y, letter, transform=ax.transAxes, fontsize=17, fontweight="bold", va="bottom", ha="left")
    ax.text(x_title, y, title, transform=ax.transAxes, fontsize=12, va="bottom", ha="left", **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/shp/final_version_no_missing_values.shp")
    ap.add_argument("--out", default="figures/regions_plot.pdf")
    a = ap.parse_args()

    gdf = gpd.read_file(a.data, ignore_geometry=True)
    gdf["region"] = gdf["SIGLA_UF"].map(UF_REGION)
    gdf = gdf.dropna(subset=["region", "cluster"]).copy()
    gdf["cluster"] = gdf["cluster"].astype(int)

    states = gdf.groupby(["CD_UF", "NM_UF", "region"], as_index=False)[["wtr_idx", "sanit_idx"]].mean()
    counts = gdf.groupby(["region", "cluster"]).size().unstack(fill_value=0).reindex(REGIONS)
    perc = counts.div(counts.sum(axis=1), axis=0) * 100

    grid = "#dddddd"
    fig = plt.figure(figsize=(9.3, 10.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.15, 1.0], width_ratios=[2.35, 1.0], hspace=0.2, wspace=0.3,
                          left=0.08, right=0.97, top=0.925, bottom=0.03)

    # (a) scatter by state (circles only, labels below points, as in the original)
    ax = fig.add_subplot(gs[0, 0])
    for r in REGIONS:
        d = states[states["region"] == r]
        ax.scatter(d["wtr_idx"], d["sanit_idx"], s=95, color=REGION_COLORS[r], alpha=0.85, edgecolor="white",
                   linewidth=0.6, label=r, zorder=2)
    for row in states.itertuples():
        ax.annotate(row.NM_UF, (row.wtr_idx, row.sanit_idx), xytext=(0, -9), textcoords="offset points",
                    ha="center", va="top", fontsize=7.5)
    ax.set_xlabel("Water Index", fontsize=11); ax.set_ylabel("Sanitation Index", fontsize=11)
    ax.grid(True, color=grid, lw=0.8); ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(grid)
    ax.tick_params(labelsize=9, color=grid)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    panel_title(ax, "a", "State-level water and sanitation performance")

    # (b) stacked bars; legend over the bars (inside the axes, lower part), as in the original
    bx = fig.add_subplot(gs[0, 1])
    bottom = np.zeros(len(REGIONS))
    for k in sorted(perc.columns):
        bx.bar(np.arange(len(REGIONS)), perc[k].values, bottom=bottom, width=0.5, color=REGIME_COLORS[k],
               edgecolor=REGIME_EDGE, linewidth=0.7, label=REGIME_NAMES[k])
        bottom += perc[k].values
    bx.set_xticks(np.arange(len(REGIONS))); bx.set_xticklabels(REGIONS, rotation=30, ha="right", fontsize=9)
    bx.set_ylim(0, 100); bx.set_ylabel("% in cluster", fontsize=11)
    bx.tick_params(axis="y", labelsize=9, color=grid)
    bx.grid(True, axis="y", color=grid, lw=0.8); bx.set_axisbelow(True)
    for sp in bx.spines.values():
        sp.set_color(grid)
    h, l = bx.get_legend_handles_labels()
    bx.legend(h, l, fontsize=8, loc="lower right", framealpha=0.95)
    bx.text(-0.02, 1.03, "b", transform=bx.transAxes, fontsize=17, fontweight="bold", va="bottom", ha="right")
    bx.text(0.5, 1.03, "Regional distribution of\ninfrastructural regimes", transform=bx.transAxes, fontsize=12,
            va="bottom", ha="center")

    # (c) Sankey, full width, shorter
    cx = fig.add_subplot(gs[1, :])
    sankey(cx, counts)
    panel_title(cx, "c", "Regional contributions to infrastructural regimes", x_title=0.03)

    fig.savefig(a.out, dpi=300)
    print("saved", a.out)


if __name__ == "__main__":
    main()
