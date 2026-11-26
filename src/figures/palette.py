"""Shared palette for the three regimes (Okabe-Ito, colour-blind-safe, categorical).

Same colours as the QGIS map composition (Fig. 4); reuse in Figs. 2, 3 and 8b/c so
each regime has the same colour everywhere. (Fig. 1 shows continuous indices and
keeps a sequential Blues scale instead.)
Keys follow the cluster ids of `cluster_3` in dataset/clusters/clusters.shp
(0 = High-coverage, 1 = Partial-coverage, 2 = Low-coverage).
"""
REGIME_NAMES = {0: "High-coverage", 1: "Partial-coverage", 2: "Low-coverage"}
REGIME_COLORS = {0: "#0072B2", 1: "#F0E442", 2: "#D55E00"}
REGIME_EDGE = "#333333"  # thin dark outline, keeps the yellow visible on white
