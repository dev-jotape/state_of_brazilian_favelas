"""Shared palette for the three regimes (colour-blind-safe sequential blues).

Reuse in Figs. 2, 3, 4 and 8b/c so the regimes look the same everywhere.
Keys follow the cluster ids of `cluster_3` in dataset/clusters/clusters.shp
(0 = High-coverage, 1 = Partial-coverage, 2 = Low-coverage).
"""
REGIME_NAMES = {0: "High-coverage", 1: "Partial-coverage", 2: "Low-coverage"}
REGIME_COLORS = {0: "#08519C", 1: "#4292C6", 2: "#C6DBEF"}
REGIME_EDGE = "#08306B"  # thin outline, keeps the lightest blue visible on white
