"""Methods figures: (1) elbow + silhouette for k=2..5, (2) t-SNE projection coloured by regime,
(3) national distribution of the four indices. Vector PDFs with editable text (fonttype 42).

Colours: regimes in the t-SNE use the shared Okabe-Ito palette (palette.py). Elbow and index histograms are not
about regimes, so they use neutral colours + different line styles/markers (no red/green pairs).

Usage (repo root; --data needs wtr_idx, sanit_idx, liv_a_idx, hh_idx [, cluster]):
    python src/figures/fig_methods_clusters.py --data dataset/shp/final_version_no_missing_values.shp --outdir figures
Use --only elbow|tsne|dist to run a single figure.
"""
import argparse
import os
import sys

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(__file__))
from palette import REGIME_COLORS, REGIME_NAMES  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

COLS = ["wtr_idx", "sanit_idx", "liv_a_idx", "hh_idx"]
TITLES = ["Improved Water", "Improved Sanitation", "Sufficient Living Area", "Durable Housing"]
GREY = "#333333"
SILH = "#009E73"
BAR = "#C8C8C8"


def elbow(X, out):
    K = list(range(2, 6))
    inertia, sil = [], []
    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        inertia.append(km.inertia_); sil.append(silhouette_score(X, km.labels_))
        print(f"k={k}: inertia={inertia[-1]:.2f}  silhouette={sil[-1]:.3f}")
    fig, ax1 = plt.subplots(figsize=(5.6, 4.4))
    ax2 = ax1.twinx()
    l1 = ax1.plot(K, inertia, marker="o", color=GREY, label="Inertia (Elbow Method)")
    ax1.set_ylabel("Inertia (Sum of squared distances)", color=GREY)
    ax1.tick_params(axis="y", labelcolor=GREY)
    l2 = ax2.plot(K, sil, marker="s", ls="--", color=SILH, label="Silhouette Score")
    ax2.set_ylabel("Silhouette Score", color=SILH)
    ax2.tick_params(axis="y", labelcolor=SILH)
    l3 = ax2.axvline(3, ls=":", color="black", lw=1.5, label="Selected k=3")
    ax1.set_xlabel("Number of clusters"); ax1.set_xticks(K)
    hs = l1 + l2 + [l3]
    ax1.legend(hs, [h.get_label() for h in hs], loc="upper right", fontsize=8.5)
    ax1.grid(True, ls="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(out); plt.close(fig); print("saved", out)


def tsne(X, labels, out):
    xy = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X)
    order = np.random.RandomState(0).permutation(len(xy))      # avoid one regime hiding the others
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.scatter(xy[order, 0], xy[order, 1], c=[REGIME_COLORS[int(l)] for l in labels[order]], s=9, alpha=0.6,
               linewidths=0, rasterized=True)
    ax.set_xlabel("t-SNE Component 1"); ax.set_ylabel("t-SNE Component 2")
    hs = [plt.Line2D([], [], marker="o", ls="", color=REGIME_COLORS[c], markersize=8, label=REGIME_NAMES[c]) for c in (0, 1, 2)]
    ax.legend(handles=hs, title="Regime", fontsize=8.5, title_fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig); print("saved", out)


def dist(df, out):
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 4.8))
    for ax, c, t in zip(axes.ravel(), COLS, TITLES):
        v = df[c].values
        ax.hist(v, bins=30, color=BAR, edgecolor="#555555", linewidth=0.5)
        # kernel density curve scaled to counts, as in the original
        from scipy.stats import gaussian_kde
        xs = np.linspace(v.min(), v.max(), 300)
        bw = (v.max() - v.min()) / 30
        ax.plot(xs, gaussian_kde(v)(xs) * len(v) * bw, color=GREY, lw=1.4)
        ax.axvline(v.mean(), ls="--", color="black", lw=1.4, label=f"Mean: {v.mean():.2f}")
        ax.axvline(np.median(v), ls=":", color=SILH, lw=1.8, label=f"Median: {np.median(v):.2f}")
        ax.set_title(t, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Index Value", fontsize=9); ax.set_ylabel("Frequency", fontsize=9)
        ax.tick_params(labelsize=8); ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.25); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out); plt.close(fig); print("saved", out, "n =", len(df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/shp/final_version_no_missing_values.shp")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--only", choices=["elbow", "tsne", "dist"])
    a = ap.parse_args()
    df = gpd.read_file(a.data, ignore_geometry=True).dropna(subset=COLS)
    X = df[COLS].values
    print("n =", len(df))
    if a.only in (None, "elbow"):
        elbow(X, os.path.join(a.outdir, "elbow_silhouette.pdf"))
    if a.only in (None, "dist"):
        dist(df, os.path.join(a.outdir, "index_dist.pdf"))
    if a.only in (None, "tsne"):
        tsne(X, df["cluster"].values, os.path.join(a.outdir, "t-SNE.pdf"))


if __name__ == "__main__":
    main()
