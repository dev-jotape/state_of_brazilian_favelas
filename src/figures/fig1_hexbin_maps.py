"""Recreate Fig. 1 (hexbin maps of the four favela-level indices).

Each hexagon shows the mean index value of the favela polygons whose centroid
falls inside it; cells with no favela are drawn as empty grey outlines.
The colour palette is a parameter (--cmap), so it can be swapped later for a
colour-blind-safe sequential scale (e.g. Blues) without touching the logic.

Usage (from the repository root):
    python src/figures/fig1_hexbin_maps.py \
        --clusters dataset/clusters/clusters.shp \
        --municipalities dataset/br_municipalities_2022/BR_Municipios_2022.shp \
        --out figures/hexbin_map_recreated.png
    # colour-blind-safe sequential blues (R1.8):
    python src/figures/fig1_hexbin_maps.py --cmap Blues --cmap-min 0.12 --out figures/hexbin_map.pdf
    # the journal needs vector (PDF/EPS), >=300 dpi RGB; use a .png --out for a preview.
"""
import argparse

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42  # editable (TrueType) text in PDF/EPS
matplotlib.rcParams["ps.fonttype"] = 42

INDICES = [  # (column in clusters.shp, panel title)
    ("wtr_idx", "Improved Water Index"),
    ("sanit_idx", "Improved Sanitation Index"),
    ("living_are", "Sufficient Living Area Index"),
    ("durable_ho", "Durable Housing Index"),
]
DEFAULT_CMAP = "RdYlGn"  # current palette of the article; change here or via --cmap
XLIM, YLIM = (-74.5, -28.5), (-35.0, 6.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", default="dataset/clusters/clusters.shp")
    ap.add_argument("--municipalities", default="dataset/br_municipalities_2022/BR_Municipios_2022.shp")
    ap.add_argument("--out", default="figures/hexbin_map_recreated.png")
    ap.add_argument("--cmap", default=DEFAULT_CMAP)
    ap.add_argument("--cmap-min", type=float, default=0.0,
                    help="start of the colormap range (0-1); e.g. 0.12 for Blues so the lowest "
                         "values stay distinguishable from the white empty cells")
    ap.add_argument("--gridsize", type=int, default=60)
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()

    gdf = gpd.read_file(a.clusters).to_crs(4674)
    cent = gdf.geometry.centroid  # lon/lat of each favela polygon
    x, y = cent.x.values, cent.y.values

    muni = gpd.read_file(a.municipalities, columns=["CD_MUN", "geometry"]).to_crs(4674)
    muni["uf"] = muni["CD_MUN"].str[:2]
    states = muni.dissolve(by="uf")
    states["geometry"] = states.geometry.simplify(0.01)

    cmap = plt.get_cmap(a.cmap)
    if a.cmap_min > 0:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            a.cmap + "_trunc", cmap(np.linspace(a.cmap_min, 1, 256)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    hb = None
    for ax, (col, title) in zip(axes.ravel(), INDICES):
        # background grid of empty cells
        ax.hexbin(x, y, gridsize=a.gridsize, extent=XLIM + YLIM, mincnt=None,
                  facecolor="none", edgecolor="lightgrey", linewidths=0.3, alpha=0.6)
        hb = ax.hexbin(x, y, C=gdf[col].values, reduce_C_function=np.mean,
                       gridsize=a.gridsize, extent=XLIM + YLIM, mincnt=1,
                       cmap=cmap, vmin=0, vmax=1, linewidths=0.2, edgecolors="face")
        states.boundary.plot(ax=ax, color="black", linewidth=0.4)
        ax.set_xlim(XLIM); ax.set_ylim(YLIM)
        ax.set_title(title)
        ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
        ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    cax = fig.add_axes([0.93, 0.05, 0.015, 0.9])
    fig.colorbar(hb, cax=cax)
    # vector output (PDF/EPS) for the journal; PNG only as a preview
    fig.savefig(a.out, dpi=a.dpi)
    print("saved", a.out)


if __name__ == "__main__":
    main()
