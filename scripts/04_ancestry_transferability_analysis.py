#!/usr/bin/env python3
"""
04_ancestry_transferability_analysis.py

A well-documented limitation of nearly every published polygenic score is
that it was developed from GWAS conducted predominantly (or, as with
PGS000055, entirely) in African American and/or European ancestry cohorts,
then often applied to other ancestries without validation. This script uses
the *real* PRS values computed in scripts/03 (no simulated data anywhere
here) plus the *real* super-population labels from 1000 Genomes to show
that effect directly: mean PRS differs systematically by ancestry, which
reflects allele-frequency differences at the scored SNPs between
populations, not necessarily a real difference in colorectal cancer risk.

This matters for the ML modeling in scripts/06: any classifier trained on
one ancestry's PRS distribution should be expected to mis-calibrate if
applied to another, which is exactly the real-world PRS clinical-utility
problem discussed in the README.

Usage:
    python scripts/04_ancestry_transferability_analysis.py \
        --prs results/prs_scores.csv \
        --output-dir results \
        --figures-dir figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

SUPER_POP_ORDER = ["AFR", "AMR", "EAS", "EUR", "SAS"]
SUPER_POP_LABELS = {
    "AFR": "African",
    "AMR": "Admixed American",
    "EAS": "East Asian",
    "EUR": "European",
    "SAS": "South Asian",
}


def summarize_prs_by_superpopulation(df: pd.DataFrame) -> pd.DataFrame:
    """Return count/mean/std/min/max of prs_raw grouped by super_pop."""
    return (
        df.groupby("super_pop")["prs_raw"]
        .agg(["count", "mean", "std", "min", "max"])
        .reindex(SUPER_POP_ORDER)
        .round(4)
    )


def run_ancestry_anova(df: pd.DataFrame):
    """One-way ANOVA of mean prs_raw across the five super-populations.

    Returns (f_statistic, p_value). A significant difference is expected from
    allele-frequency differences at the scored SNPs, not true CRC risk.
    """
    groups = [df.loc[df["super_pop"] == sp, "prs_raw"] for sp in SUPER_POP_ORDER]
    return stats.f_oneway(*groups)


def write_anova_report(path: str, f_stat: float, p_value: float) -> None:
    """Write ANOVA F/p and a short interpretation note to ``path``."""
    with open(path, "w") as fh:
        fh.write(
            "One-way ANOVA: mean PRS (PGS000055, colorectal cancer) across "
            "1000 Genomes super-populations\n"
            f"F = {f_stat:.3f}, p = {p_value:.3e}\n\n"
            "PGS000055 was trained on African American + European ancestry "
            "GWAS. A significant difference in mean PRS across all five 1000 "
            "Genomes super-populations is expected here purely from "
            "allele-frequency differences at the scored SNPs, and is a "
            "textbook illustration of why polygenic scores generally do not "
            "transfer across ancestries without recalibration -- it is not "
            "evidence of a real difference in colorectal cancer risk.\n"
        )


def plot_prs_by_superpopulation(df: pd.DataFrame, figures_dir: str) -> str:
    """Save box+strip plot of prs_raw by super_pop; return the figure path."""
    order = [sp for sp in SUPER_POP_ORDER if sp in df["super_pop"].unique()]
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df, x="super_pop", y="prs_raw", hue="super_pop",
        order=order, palette="Set2", legend=False,
    )
    sns.stripplot(data=df, x="super_pop", y="prs_raw", order=order, color="black", alpha=0.15, size=2)
    plt.xlabel("1000 Genomes super-population")
    plt.ylabel("Colorectal cancer PRS (PGS000055, raw)")
    plt.title("A Euro/African-American-derived CRC PRS does not transfer evenly across ancestries")
    plt.xticks(range(len(order)), [f"{sp}\n({SUPER_POP_LABELS[sp]})" for sp in order])
    plt.tight_layout()
    out_path = os.path.join(figures_dir, "07_prs_by_superpopulation.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main():
    """Summarize PRS by ancestry, run ANOVA, and write figure 07."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figures-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    df = pd.read_csv(args.prs)

    summary = summarize_prs_by_superpopulation(df)
    summary.to_csv(os.path.join(args.output_dir, "prs_by_superpopulation.csv"))
    print(summary)

    f_stat, p_value = run_ancestry_anova(df)
    write_anova_report(os.path.join(args.output_dir, "ancestry_anova.txt"), f_stat, p_value)
    print(f"\nANOVA: F={f_stat:.3f}, p={p_value:.3e}")

    fig_path = plot_prs_by_superpopulation(df, args.figures_dir)
    print(f"Wrote figure: {fig_path}")


if __name__ == "__main__":
    main()
