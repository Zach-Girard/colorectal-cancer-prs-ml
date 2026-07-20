#!/usr/bin/env bash
#
# run_all.sh
#
# Runs the full pipeline end to end: download real PGS weights + real 1000
# Genomes genotypes, compute real per-individual PRS, run the (real-data,
# no-simulation) ancestry-transferability analysis, simulate a documented
# case/control label, then train and evaluate the ML classifiers.
#
# Usage: scripts/run_all.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== 1/6: Downloading PGS Catalog weights (PGS000055, colorectal cancer) ==="
bash scripts/01_download_pgs_weights.sh data/pgs_catalog

echo
echo "=== 2/6: Extracting matching genotypes from 1000 Genomes Phase 3 ==="
bash scripts/02_extract_1000genomes_genotypes.sh \
  data/pgs_catalog/PGS000055_hmPOS_GRCh37.txt.gz \
  data/1000genomes

echo
echo "=== 3/6: Computing per-individual PRS ==="
python3 scripts/03_compute_prs.py \
  --weights data/pgs_catalog/PGS000055_hmPOS_GRCh37.txt.gz \
  --vcf data/1000genomes/pgs000055_snps.vcf \
  --panel data/1000genomes/sample_populations.panel \
  --output results/prs_scores.csv

echo
echo "=== 4/6: Ancestry-transferability analysis (real data only) ==="
python3 scripts/04_ancestry_transferability_analysis.py \
  --prs results/prs_scores.csv \
  --output-dir results \
  --figures-dir figures

echo
echo "=== 5/6: Simulating documented case/control phenotype ==="
python3 scripts/05_simulate_phenotype.py \
  --prs results/prs_scores.csv \
  --output results/simulated_cohort.csv \
  --prevalence 0.045 \
  --seed 42

echo
echo "=== 6/6: Training and evaluating ML classifiers ==="
python3 scripts/06_train_evaluate_models.py \
  --cohort results/simulated_cohort.csv \
  --output-dir results \
  --figures-dir figures

echo
echo "Done. Figures in figures/, tables in results/."
