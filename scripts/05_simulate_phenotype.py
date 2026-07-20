#!/usr/bin/env python3
"""
05_simulate_phenotype.py

*** This script is the one place in the pipeline that introduces simulated
    data, and it is documented here and in the README as exactly that. ***

The 1000 Genomes Project (used for real genotypes/PRS in scripts/03) and
the PGS Catalog (used for real GWAS weights in scripts/01) do not, and
should not, carry individual-level disease labels -- that kind of
genotype-phenotype linkage is only available through controlled-access
resources like UK Biobank or dbGaP, which cannot be redistributed in a
public repo. To demonstrate the downstream ML workflow (scripts/06)
end-to-end, this script layers a simulated colorectal cancer (CRC) case/
control label on top of:

  - REAL per-person PRS (scripts/03, computed from real genotypes + real
    published GWAS weights)
  - REAL sex (1000 Genomes `gender` field)
  - SIMULATED age (drawn independently of any real attribute)
  - SIMULATED family history of CRC (drawn independently)

via a logistic risk model:

    logit(P(case)) = beta0 + beta_prs * PRS_z + beta_age * age_z
                      + beta_family_history * family_history

`BETA_PRS` below was not picked arbitrarily. I ran a small calibration
sweep beforehand -- trying candidate values in a large synthetic sample and
checking what AUC a PRS-only logistic model would achieve for each one --
then kept the value that reproduced the *actually published* AUROC for
PGS000055 (0.65, 95% CI 0.62-0.69; Schmit et al. 2019, evaluated in an
independent EPIC/UK Biobank/MGI cohort). That anchors the simulated effect
size to real, reported evidence rather than a number that merely "looked
reasonable." `beta0` (the intercept) is calibrated separately, via the
bisection search below, so the overall simulated prevalence matches a
realistic population-level CRC prevalence rather than an arbitrary one.

Usage:
    python scripts/05_simulate_phenotype.py \
        --prs results/prs_scores.csv \
        --output results/simulated_cohort.csv \
        --prevalence 0.045 \
        --seed 42
"""
import argparse
import os

import numpy as np
import pandas as pd

# Calibrated so a PRS-only logistic model on this label reproduces
# PGS000055's published AUROC of ~0.65 (see docstring above).
BETA_PRS = 0.58
# Age is one of the strongest known CRC risk factors; family history
# roughly doubles risk in the epidemiological literature (OR ~1.8-2.2) --
# beta_family_history = 0.65 corresponds to an OR of ~1.9 at fixed PRS/age.
BETA_AGE = 0.35
BETA_FAMILY_HISTORY = 0.65


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calibrate_intercept(linear_predictor: np.ndarray, target_prevalence: float) -> float:
    """Bisection search for the intercept that makes mean(P(case)) equal
    to the target prevalence, given everyone's linear predictor without
    the intercept term."""
    lo, hi = -15.0, 15.0
    for _ in range(100):
        mid = (lo + hi) / 2
        mean_p = sigmoid(mid + linear_predictor).mean()
        if mean_p < target_prevalence:
            lo = mid
        else:
            hi = mid
    return mid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prevalence", type=float, default=0.045,
                         help="Target simulated CRC prevalence [default: %(default)s, "
                              "approximating US lifetime risk]")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.prs)

    n = len(df)
    df["age"] = np.clip(rng.normal(60, 10, n), 40, 90).round(1)
    df["age_z"] = (df["age"] - df["age"].mean()) / df["age"].std()
    df["family_history"] = rng.binomial(1, 0.10, n)
    df["sex"] = df["gender"]  # real 1000 Genomes attribute, not simulated

    linear_predictor = (
        BETA_PRS * df["prs_z"]
        + BETA_AGE * df["age_z"]
        + BETA_FAMILY_HISTORY * df["family_history"]
    )
    intercept = calibrate_intercept(linear_predictor.values, args.prevalence)
    p_case = sigmoid(intercept + linear_predictor)
    df["case"] = rng.binomial(1, p_case)

    realized_prevalence = df["case"].mean()
    print(f"Simulated {n} individuals: {df['case'].sum()} cases ({realized_prevalence:.1%}), "
          f"target was {args.prevalence:.1%} (intercept={intercept:.3f})")

    cols = ["sample_id", "super_pop", "pop", "sex", "prs_raw", "prs_z",
            "age", "family_history", "case"]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df[cols].to_csv(args.output, index=False)
    print(f"Wrote simulated cohort to {args.output}")


if __name__ == "__main__":
    main()
