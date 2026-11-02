"""Render SHAP importance CSVs without loading patient data or retraining models.

Example:
    python src/05_ml/plot_shap_importance.py --csv results/ml/Chemotherapy_shap_importance.csv --therapy Chemotherapy --out-dir results/ml
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from feature_schema import is_reaction_feature


def classify_features(frame):
    required = {"feature", "mean_abs_shap"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing SHAP columns: {sorted(missing)}")
    frame = frame.copy()
    frame["mean_abs_shap"] = pd.to_numeric(frame["mean_abs_shap"], errors="raise")
    if frame["mean_abs_shap"].isna().any() or (frame["mean_abs_shap"] < 0).any():
        raise ValueError("Mean absolute SHAP values must be nonnegative and nonmissing")
    frame["feature_type"] = frame["feature"].map(
        lambda name: "metabolic_flux" if is_reaction_feature(name) else "clinical"
    )
    frame = frame.sort_values("mean_abs_shap", ascending=False, kind="stable").reset_index(drop=True)
    frame["rank"] = frame.index + 1
    return frame


def plot_shap_importance(frame, therapy, out_dir, top_n=30):
    """Correct feature colours from names; preserve the supplied numeric values."""
    if top_n < 1:
        raise ValueError("top_n must be positive")
    frame = classify_features(frame)
    shown = frame.head(top_n)
    if shown.empty:
        raise ValueError("No SHAP features to plot")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = therapy.replace(" ", "_")
    colours = shown["feature_type"].map({"metabolic_flux": "steelblue", "clinical": "darkorange"})
    fig, ax = plt.subplots(figsize=(10, max(5, len(shown) * 0.29)))
    ax.barh(range(len(shown)), shown["mean_abs_shap"], color=colours, alpha=0.8)
    ax.set_yticks(range(len(shown)), labels=shown["feature"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute SHAP value", fontsize=11)
    ax.set_title(f"Feature contributions to {therapy.lower()} response prediction", fontsize=12)
    ax.legend(handles=[Patch(color="steelblue", alpha=0.8, label="Metabolic reaction"),
                       Patch(color="darkorange", alpha=0.8, label="Clinical")], fontsize=9)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(out_dir / f"{prefix}_shap_bar.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    frame.to_csv(out_dir / f"{prefix}_shap_importance.csv", index=False)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--therapy", required=True)
    parser.add_argument("--out-dir", default="results/ml")
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()
    plot_shap_importance(pd.read_csv(args.csv), args.therapy, args.out_dir, args.top_n)


if __name__ == "__main__":
    main()
