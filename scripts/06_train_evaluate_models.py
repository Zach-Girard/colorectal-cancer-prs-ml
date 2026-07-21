#!/usr/bin/env python3
"""
06_train_evaluate_models.py

Trains and evaluates classifiers that predict the simulated colorectal
cancer case/control label (scripts/05) from the real PRS plus a mix of
real (sex) and simulated (age, family history) covariates, then produces
the standard set of figures used to validate a polygenic score in the
literature: ROC curves, a calibration curve, and a risk-decile
stratification plot (relative odds of being a case in each PRS decile vs.
the middle decile).

Three models are compared:
  1. Logistic regression, PRS only         -- the "clinical baseline"
  2. Logistic regression, PRS + covariates -- age, family history, sex
  3. Gradient boosting, PRS + covariates   -- can capture nonlinearities/
                                               interactions the linear
                                               models can't

Because there are only ~100 simulated cases in a cohort of 2,504, AUC is
estimated via 5-fold stratified cross-validation (mean +/- SD across
folds) rather than a single train/test split, and out-of-fold predictions
(via cross_val_predict) are used for the calibration and decile plots so
those diagnostics aren't evaluated on the same rows used to fit the model.

Usage:
    python scripts/06_train_evaluate_models.py \
        --cohort results/simulated_cohort.csv \
        --output-dir results \
        --figures-dir figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score


def evaluate_model(model, X, y, cv, model_name):
    """Run stratified CV: return per-fold AUCs and out-of-fold P(case).

    Uses cross_val_score for fold AUCs and cross_val_predict for oof
    probabilities (ROC / calibration plots).
    """
    aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    oof_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    print(f"{model_name}: AUC = {aucs.mean():.3f} +/- {aucs.std():.3f} (5-fold CV)")
    return aucs, oof_proba


def odds_ratio_per_sd(model, feature_names, feature="prs_z"):
    """Return exp(coef) for ``feature`` from a fitted logistic regression.

    For z-scored features this is the OR per 1 SD increase.
    """
    idx = feature_names.index(feature)
    beta = model.coef_[0][idx]
    return np.exp(beta)


def main():
    """Train/evaluate models; write metrics and figures 08–10."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figures-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    df = pd.read_csv(args.cohort)
    df["sex_male"] = (df["sex"] == "male").astype(int)
    df["age_z"] = (df["age"] - df["age"].mean()) / df["age"].std()

    y = df["case"].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    feature_sets = {
        "PRS only": ["prs_z"],
        "PRS + covariates (LR)": ["prs_z", "age_z", "family_history", "sex_male"],
    }

    results = {}

    lr_prs_only = LogisticRegression()
    results["PRS only"] = evaluate_model(
        lr_prs_only, df[feature_sets["PRS only"]].values, y, cv, "Logistic regression, PRS only"
    )

    lr_full = LogisticRegression()
    results["PRS + covariates (LR)"] = evaluate_model(
        lr_full, df[feature_sets["PRS + covariates (LR)"]].values, y, cv, "Logistic regression, PRS + covariates"
    )

    gb_full = GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.1, random_state=args.seed)
    results["PRS + covariates (GB)"] = evaluate_model(
        gb_full, df[feature_sets["PRS + covariates (LR)"]].values, y, cv, "Gradient boosting, PRS + covariates"
    )

    # Full-data refit for an interpretable OR only — not used for CV AUC.
    lr_full_fit = LogisticRegression().fit(df[feature_sets["PRS + covariates (LR)"]].values, y)
    or_per_sd = odds_ratio_per_sd(lr_full_fit, feature_sets["PRS + covariates (LR)"], "prs_z")

    # --- Metrics table -------------------------------------------------
    metrics_rows = []
    for name, (aucs, _) in results.items():
        metrics_rows.append({"model": name, "auc_mean": aucs.mean(), "auc_std": aucs.std()})
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(args.output_dir, "model_performance.csv"), index=False)
    print(metrics_df)
    print(f"\nOdds ratio per 1-SD increase in PRS (PRS + covariates model): {or_per_sd:.2f}")
    with open(os.path.join(args.output_dir, "model_performance.csv"), "a") as fh:
        fh.write(f"\n# Odds ratio per 1-SD PRS increase (PRS+covariates LR): {or_per_sd:.3f}\n")

    # --- Figure: ROC curves (out-of-fold predictions) -------------------
    plt.figure(figsize=(6, 6))
    for name, (aucs, oof_proba) in results.items():
        fpr, tpr, _ = roc_curve(y, oof_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={aucs.mean():.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Out-of-fold ROC: predicting simulated CRC case status")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, "08_roc_curves.png"), dpi=150)
    plt.close()

    # --- Figure: calibration curve (highest-AUC model) ------------------
    best_name = metrics_df.loc[metrics_df["auc_mean"].idxmax(), "model"]
    _, best_oof_proba = results[best_name]
    frac_pos, mean_pred = calibration_curve(y, best_oof_proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot(mean_pred, frac_pos, "o-", label=best_name)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    plt.xlabel("Mean predicted risk (decile bin)")
    plt.ylabel("Observed case fraction (decile bin)")
    plt.title(f"Calibration: {best_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, "09_calibration_curve.png"), dpi=150)
    plt.close()

    # --- Figure: case rate by PRS decile vs middle deciles ---------------
    df["prs_decile"] = pd.qcut(df["prs_z"], 10, labels=False) + 1
    decile_rates = df.groupby("prs_decile")["case"].mean()
    middle_decile_rate = decile_rates.loc[[5, 6]].mean()
    decile_or = decile_rates / middle_decile_rate

    decile_or.to_csv(os.path.join(args.output_dir, "prs_decile_odds_ratios.csv"), header=["odds_ratio_vs_middle_decile"])

    plt.figure(figsize=(8, 5))
    plt.bar(decile_or.index.astype(str), decile_or.values, color="steelblue")
    plt.axhline(1.0, color="black", linestyle="--", alpha=0.6, label="Middle-decile (5th-6th) risk")
    plt.xlabel("PRS decile (1 = lowest genetic risk, 10 = highest)")
    plt.ylabel("Relative odds of being a case vs. middle decile")
    plt.title("Risk stratification by colorectal cancer PRS decile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, "10_risk_decile_stratification.png"), dpi=150)
    plt.close()

    print(f"\nTop decile vs. middle decile odds ratio: {decile_or.iloc[-1]:.2f}")
    print("Wrote figures 08-10 and result tables to", args.figures_dir, "/", args.output_dir)


if __name__ == "__main__":
    main()
