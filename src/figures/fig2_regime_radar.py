"""Recreate Fig. 2 (mean indices of the three regimes, radar charts).

Usage (from the repository root):
    python src/figures/fig2_regime_radar.py \
        --clusters dataset/clusters/clusters.shp --out figures/cluster_avg2.pdf
Drawn at 8 in and shown at the full text width (5.4 in): 9 pt here is ~6 pt in print (E.10).
Vector PDF (editable text, RGB); use a .png --out for a preview.
"""
import argparse
import os
import sys

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from palette import REGIME_COLORS, REGIME_EDGE, REGIME_NAMES  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

INDICES = ["wtr_idx", "sanit_idx", "living_are", "durable_ho"]
LABELS = ["Water", "Sanitation", "Living\nArea", "Housing"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", default="dataset/clusters/clusters.shp")
    ap.add_argument("--out", default="figures/cluster_avg2.pdf")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()

    df = gpd.read_file(a.clusters, ignore_geometry=True)
    means = df.groupby("cluster_3")[INDICES].mean()

    angles = np.linspace(0, 2 * np.pi, len(INDICES), endpoint=False).tolist()
    angles += angles[:1]
    fig, axs = plt.subplots(1, 3, figsize=(8, 3.3), subplot_kw=dict(polar=True))
    for ax, k in zip(axs, [0, 1, 2]):
        vals = means.loc[k].tolist()
        v = vals + vals[:1]
        ax.plot(angles, v, color=REGIME_EDGE, linewidth=1.0, zorder=3)
        ax.fill(angles, v, facecolor=REGIME_COLORS[k], edgecolor=REGIME_EDGE,
                linewidth=1.0, alpha=0.8, zorder=2)
        ax.scatter(angles[:-1], vals, color=REGIME_COLORS[k], edgecolor=REGIME_EDGE,
                   s=18, linewidth=0.8, zorder=4)
        for ang, val in zip(angles[:-1], vals):
            ax.text(ang, val - 0.15, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color="black", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        # same orientation as the original figure: Water right, Sanitation top
        ax.set_theta_offset(0)
        ax.set_theta_direction(1)
        ax.set_thetagrids(np.degrees(angles[:-1]), LABELS, fontsize=9)
        ax.tick_params(axis="x", pad=6)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1"], fontsize=7)
        ax.set_rlabel_position(45)
        ax.grid(color="lightgrey", linewidth=0.6)
        ax.set_title(REGIME_NAMES[k], fontsize=10, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(a.out, dpi=a.dpi)
    print("saved", a.out)


if __name__ == "__main__":
    main()
