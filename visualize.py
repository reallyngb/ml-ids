"""
visualize.py
------------
Phase 4 of the IDS pipeline: Exploratory Data Analysis.

Generates and saves (to data/processed/eda_plots/):
  1. class_distribution.png  - bar chart of BENIGN vs attack counts
     WHY: shows class imbalance, which justifies using SMOTE in Phase 3
     and Precision/Recall (not Accuracy) in Phase 11.
  2. correlation_heatmap.png - feature-feature Pearson correlation
     WHY: highly correlated features (e.g. Total Fwd Packets vs Subflow
     Fwd Bytes) are redundant; can be candidates for removal via feature
     selection in Phase 12.
  3. feature_importance.png  - Random Forest impurity-based importance
     WHY: tells us which flow statistics actually separate attacks from
     benign traffic, and sanity-checks that the model isn't relying on
     something spurious.
  4. boxplots.png            - top features, benign vs attack
     WHY: visually confirms the separability the model will exploit, and
     flags outliers per class.
  5. histograms.png          - distribution of top features
     WHY: skewed distributions (heavy right tail) are common in network
     flow data (most flows are short, a few are huge) and explain why
     scaling is necessary before distance-based models (KNN, SVM).
  6. pca_visualization.png   - 2D PCA projection colored by class
     WHY: a quick sanity check for whether classes are linearly separable
     in a reduced-dimension view — helps set expectations for model choice.

Run directly: `python src/eda/visualize.py` (uses the cleaned data straight
from disk, before scaling/SMOTE, so plots reflect real-world distributions).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display in this environment; write straight to file
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config_loader import get_config, resolve_path
from src.utils.logger import get_logger
from src.data.preprocessing import load_raw_data, clean_data, encode_labels

logger = get_logger(__name__)
sns.set_theme(style="whitegrid")


def plot_class_distribution(df, cfg, out_dir):
    plt.figure(figsize=(7, 5))
    order = df[cfg["data"]["label_column"]].value_counts().index
    sns.countplot(data=df, x=cfg["data"]["label_column"], order=order, hue=cfg["data"]["label_column"], legend=False, palette="viridis")
    plt.title("Class Distribution (raw counts)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution.png", dpi=120)
    plt.close()
    logger.info("Saved class_distribution.png")


def plot_correlation_heatmap(X: pd.DataFrame, out_dir):
    plt.figure(figsize=(14, 12))
    corr = X.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": 0.7})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png", dpi=120)
    plt.close()
    logger.info("Saved correlation_heatmap.png")


def plot_feature_importance(X: pd.DataFrame, y: pd.Series, cfg, out_dir, top_n=15):
    rf = RandomForestClassifier(n_estimators=100, random_state=cfg["project"]["random_seed"], n_jobs=-1)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(8, 6))
    importances.head(top_n).plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=120)
    plt.close()
    logger.info("Saved feature_importance.png")
    return importances


def plot_boxplots_histograms(X: pd.DataFrame, y: pd.Series, top_features, out_dir):
    df_plot = X[top_features].copy()
    df_plot["class"] = y.map({0: "BENIGN", 1: "ATTACK"})

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, feat in zip(axes.flat, top_features[:6]):
        sns.boxplot(data=df_plot, x="class", y=feat, hue="class", legend=False, ax=ax, palette="Set2")
        ax.set_title(feat, fontsize=10)
    plt.suptitle("Boxplots: Top Features by Class (outliers visible as points beyond whiskers)")
    plt.tight_layout()
    plt.savefig(out_dir / "boxplots.png", dpi=120)
    plt.close()
    logger.info("Saved boxplots.png")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, feat in zip(axes.flat, top_features[:6]):
        sns.histplot(data=df_plot, x=feat, hue="class", kde=True, ax=ax, element="step", stat="density", common_norm=False)
        ax.set_title(feat, fontsize=10)
    plt.suptitle("Histograms: Top Features by Class")
    plt.tight_layout()
    plt.savefig(out_dir / "histograms.png", dpi=120)
    plt.close()
    logger.info("Saved histograms.png")


def plot_pca(X: pd.DataFrame, y: pd.Series, cfg, out_dir):
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=cfg["project"]["random_seed"])
    components = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(components[:, 0], components[:, 1], c=y, cmap="coolwarm", alpha=0.4, s=8)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.title("PCA Projection: BENIGN (0) vs ATTACK (1)")
    plt.colorbar(scatter, label="class")
    plt.tight_layout()
    plt.savefig(out_dir / "pca_visualization.png", dpi=120)
    plt.close()
    logger.info("Saved pca_visualization.png")


def run_eda():
    cfg = get_config()
    out_dir = resolve_path(cfg["paths"]["eda_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)  # eda_output_dir is itself a directory

    df = load_raw_data(cfg)
    df_clean = clean_data(df, cfg)
    X, y, class_names, _ = encode_labels(df_clean, cfg, binary=True)

    plot_class_distribution(df_clean, cfg, out_dir)
    plot_correlation_heatmap(X, out_dir)
    importances = plot_feature_importance(X, y, cfg, out_dir)
    top_features = importances.head(6).index.tolist()
    plot_boxplots_histograms(X, y, top_features, out_dir)
    plot_pca(X, y, cfg, out_dir)

    logger.info(f"All EDA plots saved to {out_dir}")
    return importances


if __name__ == "__main__":
    run_eda()
