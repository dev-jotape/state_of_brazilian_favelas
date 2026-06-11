"""
Bootstrapped classical classifiers over TTA-averaged EfficientNetV2-S features.

Combines the bootstrapping approach (N balanced subsamples + 95% confidence
intervals) with classical classifiers on pre-extracted TTA features.

Feature extraction runs once and is cached. Each bootstrap iteration takes
seconds (classical classifiers), so 10 iterations complete in minutes.

Dataset: all cities combined.
Experiments: 3-class (0 vs 1 vs 2) and binary (0 vs 1+2).
"""

import os
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetV2S

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# GPU memory growth (needed for feature extraction)
# ---------------------------------------------------------------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPUs available: {len(gpus)}")
else:
    print("No GPU found, using CPU.")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SHAPEFILE_PATH = Path("./dataset/morphology/components_with_favela.shp")
IMAGE_BASE_DIR = Path("./dataset/gmaps_slums")
FEATURES_DIR = Path("./dataset/morphology/output/features")
OUTPUT_DIR = Path("./dataset/morphology/output/bootstrap_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_TO_FOLDER = {
    "bh": "GMAPS_RGB_BH_2024",
    "br": "GMAPS_RGB_BR_2024",
    "pa": "GMAPS_RGB_PA_2024",
    "rj": "GMAPS_RGB_RJ_2024",
    "sp": "GMAPS_RGB_SP_2024",
    "ssa": "GMAPS_RGB_SSA_2024",
}

IMAGE_SIZE = 224
BATCH_SIZE = 64
SEED = 42
N_BOOTSTRAP = 10

np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Feature extractor (frozen EfficientNetV2-S -> GAP -> 1280-d)
# ---------------------------------------------------------------------------
def build_feature_extractor() -> keras.Model:
    base = EfficientNetV2S(
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False
    return base


def _load_single(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = tf.cast(img, tf.float32)
    return keras.applications.efficientnet_v2.preprocess_input(img)


TTA_TRANSFORMS = [
    lambda img: img,
    lambda img: tf.image.rot90(img, k=1),
    lambda img: tf.image.rot90(img, k=2),
    lambda img: tf.image.rot90(img, k=3),
    lambda img: tf.image.flip_left_right(img),
    lambda img: tf.image.flip_left_right(tf.image.rot90(img, k=1)),
    lambda img: tf.image.flip_left_right(tf.image.rot90(img, k=2)),
    lambda img: tf.image.flip_left_right(tf.image.rot90(img, k=3)),
]


def extract_features_tta(
    model: keras.Model, image_paths: np.ndarray,
) -> np.ndarray:
    """Extract TTA-averaged embeddings (8 views per image)."""
    accum = None

    for t_idx, transform_fn in enumerate(TTA_TRANSFORMS):
        print(f"    TTA view {t_idx + 1}/{len(TTA_TRANSFORMS)}...")

        def load_and_transform(path, _fn=transform_fn):
            img = _load_single(path)
            return _fn(img)

        ds = tf.data.Dataset.from_tensor_slices(image_paths)
        ds = ds.map(load_and_transform, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        parts = [model.predict(b, verbose=0) for b in ds]
        feats = np.concatenate(parts, axis=0)

        if accum is None:
            accum = feats
        else:
            accum += feats

    return accum / len(TTA_TRANSFORMS)


# ---------------------------------------------------------------------------
# Classical classifiers
# ---------------------------------------------------------------------------
def get_classifiers() -> dict:
    return {
        "SVM_rbf_pca128": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=128, random_state=SEED)),
            ("clf", SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=SEED,
            )),
        ]),
    }

# ---------------------------------------------------------------------------
# Balanced subsample (returns aligned df + features)
# ---------------------------------------------------------------------------
def balanced_subsample(
    df: pd.DataFrame, features: np.ndarray, seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.RandomState(seed)
    counts = df["label"].value_counts()
    n_min = counts.min()
    indices = []
    for cls in sorted(df["label"].unique()):
        cls_idx = df.index[df["label"] == cls].values
        sampled = rng.choice(cls_idx, size=n_min, replace=True)
        indices.extend(sampled)
    rng.shuffle(indices)
    indices = np.array(indices)
    pos = np.arange(len(df))
    pos_map = dict(zip(df.index, pos))
    feat_idx = np.array([pos_map[i] for i in indices])
    return df.loc[indices].reset_index(drop=True), features[feat_idx]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_dataframe() -> pd.DataFrame:
    print("Loading shapefile...")
    gdf = gpd.read_file(SHAPEFILE_PATH)
    matched = gdf[gdf["CD_FCU"].notna()].copy()
    print(f"  Matched components: {len(matched)}")

    matched["img_path"] = matched.apply(
        lambda r: str(
            IMAGE_BASE_DIR
            / STATE_TO_FOLDER[r["state"]]
            / f"{int(r['index'])}_1.png"
        ),
        axis=1,
    )

    existing = matched["img_path"].apply(os.path.isfile)
    n_missing = (~existing).sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} images not found, skipping them.")
        matched = matched[existing].copy()

    matched["cluster"] = matched["cluster"].astype(int)
    print(f"  Final dataset: {len(matched)} images")
    print(f"  Cluster distribution:\n{matched['cluster'].value_counts().sort_index().to_string()}")
    return matched


def extract_or_load_features(df: pd.DataFrame) -> np.ndarray:
    """Load cached TTA features or extract them."""
    tta_path = FEATURES_DIR / "all_efficientnetv2s_features_tta.npy"
    if tta_path.exists():
        print(f"\n  Loading cached TTA features: {tta_path}")
        features = np.load(tta_path)
        if features.shape[0] == len(df):
            print(f"  TTA features shape: {features.shape}")
            return features
        print("  Cache size mismatch, re-extracting...")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    print("\n  Extracting EfficientNetV2-S TTA features (8 views)...")
    extractor = build_feature_extractor()
    features = extract_features_tta(extractor, df["img_path"].values)
    np.save(tta_path, features)
    print(f"  Cached: {tta_path}")
    del extractor
    keras.backend.clear_session()
    print(f"  TTA features shape: {features.shape}")
    return features


# ---------------------------------------------------------------------------
# Bootstrap experiment runner
# ---------------------------------------------------------------------------
def run_bootstrap_features_experiment(
    name: str,
    df: pd.DataFrame,
    features: np.ndarray,
    num_classes: int,
    label_names: list[str],
):
    print(f"\n{'#'*60}")
    print(f"# Bootstrap Features Experiment: {name}")
    print(f"# N_BOOTSTRAP: {N_BOOTSTRAP}")
    print(f"# Classes: {num_classes} -> {label_names}")
    print(f"# Total samples: {len(df)}, Feature dim: {features.shape[1]}")
    print(f"# Groups: {df['CD_FCU'].nunique()}")
    print(f"# Label distribution:\n{df['label'].value_counts().sort_index().to_string()}")
    print(f"{'#'*60}\n")

    # Fixed stratified test holdout (20%)
    gss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    trainval_idx, test_idx = next(gss.split(df, df["label"]))

    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    test_features = features[test_idx]
    trainval_df = df.iloc[trainval_idx].copy().reset_index(drop=True)
    trainval_features = features[trainval_idx]

    print(f"  Fixed test set: {len(test_df)} samples")
    print(f"  Test labels:\n{test_df['label'].value_counts().sort_index().to_string()}")
    print(f"  Trainval pool: {len(trainval_df)} samples\n")

    classifier_names = list(get_classifiers().keys())
    all_iter_results = {cn: [] for cn in classifier_names}

    for b in range(N_BOOTSTRAP):
        iter_seed = SEED + b * 137

        bal_df, bal_feats = balanced_subsample(trainval_df, trainval_features, seed=iter_seed)

        print(f"\n{'='*60}")
        print(f"  Bootstrap iteration {b+1}/{N_BOOTSTRAP} (seed={iter_seed})")
        print(f"  Balanced trainval: {len(bal_df)} samples")
        print(f"  Labels: {bal_df['label'].value_counts().sort_index().to_dict()}")
        print(f"{'='*60}")

        classifiers = get_classifiers()
        X_train = bal_feats
        y_train = bal_df["label"].values
        X_test = test_features
        y_test = test_df["label"].values

        for clf_name, clf in classifiers.items():
            clf_clone = clone(clf)
            clf_clone.fit(X_train, y_train)
            preds = clf_clone.predict(X_test)

            report = classification_report(
                y_test, preds, target_names=label_names, output_dict=True,
            )
            report["confusion_matrix"] = confusion_matrix(y_test, preds).tolist()
            macro_f1 = report["macro avg"]["f1-score"]
            print(f"    {clf_name:>25}: macro F1 = {macro_f1:.4f}")

            report_path = OUTPUT_DIR / f"{name}_{clf_name}_iter{b}_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            all_iter_results[clf_name].append(report)

    # -- Aggregate results --
    print(f"\n{'#'*60}")
    print(f"# AGGREGATE RESULTS ({N_BOOTSTRAP} bootstrap iterations)")
    print(f"{'#'*60}")

    aggregate = {"n_bootstrap": N_BOOTSTRAP, "classifiers": {}}
    ranking_rows = []

    for clf_name in classifier_names:
        reports = all_iter_results[clf_name]
        macro_f1s = np.array([r["macro avg"]["f1-score"] for r in reports])

        clf_agg = {
            "macro_f1": {
                "mean": float(macro_f1s.mean()),
                "std": float(macro_f1s.std()),
                "ci_95_lower": float(np.percentile(macro_f1s, 2.5)),
                "ci_95_upper": float(np.percentile(macro_f1s, 97.5)),
                "all_values": macro_f1s.tolist(),
            },
            "per_class_f1": {},
        }

        print(f"\n  {clf_name}:")
        print(f"    Macro F1: {macro_f1s.mean():.4f} +/- {macro_f1s.std():.4f}"
              f"  CI: [{np.percentile(macro_f1s, 2.5):.4f}, {np.percentile(macro_f1s, 97.5):.4f}]")

        row = {
            "classifier": clf_name,
            "mean_macro_f1": float(macro_f1s.mean()),
            "std_macro_f1": float(macro_f1s.std()),
            "ci_95_lower": float(np.percentile(macro_f1s, 2.5)),
            "ci_95_upper": float(np.percentile(macro_f1s, 97.5)),
        }

        for ln in label_names:
            vals = np.array([r[ln]["f1-score"] for r in reports])
            clf_agg["per_class_f1"][ln] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "ci_95_lower": float(np.percentile(vals, 2.5)),
                "ci_95_upper": float(np.percentile(vals, 97.5)),
            }
            print(f"    {ln} F1: {vals.mean():.4f} +/- {vals.std():.4f}"
                  f"  CI: [{np.percentile(vals, 2.5):.4f}, {np.percentile(vals, 97.5):.4f}]")
            row[f"f1_{ln}_mean"] = float(vals.mean())
            row[f"f1_{ln}_std"] = float(vals.std())

        aggregate["classifiers"][clf_name] = clf_agg
        ranking_rows.append(row)

    agg_path = OUTPUT_DIR / f"{name}_aggregate.json"
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\n  Saved: {agg_path}")

    ranking_df = pd.DataFrame(ranking_rows).sort_values("mean_macro_f1", ascending=False)
    ranking_path = OUTPUT_DIR / f"{name}_ranking.csv"
    ranking_df.to_csv(ranking_path, index=False)

    print(f"\n{'#'*60}")
    print(f"# RANKING ({name})")
    print(f"{'#'*60}")
    print(ranking_df.to_string(index=False))
    print(f"\n  Saved: {ranking_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = prepare_dataframe()
    features = extract_or_load_features(df)

    df_3class = df.copy()
    df_3class["label"] = df_3class["cluster"]
    run_bootstrap_features_experiment(
        name="bootstrap_3class",
        df=df_3class,
        features=features,
        num_classes=3,
        label_names=["cluster_0", "cluster_1", "cluster_2"],
    )

    df_binary = df.copy()
    df_binary["label"] = (df_binary["cluster"] == 2).astype(int)
    run_bootstrap_features_experiment(
        name="bootstrap_binary_2",
        df=df_binary,
        features=features,
        num_classes=2,
        label_names=["cluster_0+1", "cluster_2"],
    )


if __name__ == "__main__":
    main()
