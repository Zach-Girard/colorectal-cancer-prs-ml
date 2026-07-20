# Colorectal Cancer Polygenic Risk Score + ML

[![CI](https://github.com/Zach-Girard/colorectal-cancer-prs-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/Zach-Girard/colorectal-cancer-prs-ml/actions/workflows/ci.yml)

Computes a real polygenic risk score (PRS) for colorectal cancer from real
GWAS effect weights and real individual genotypes, uses it to demonstrate a
well-known and clinically important PRS limitation (poor transferability
across ancestries), and then trains/evaluates ML classifiers that combine
the PRS with clinical covariates to predict cancer risk.

## A note on the data, up front

This is the most important thing to understand about this repo, so it's
here before anything else:

**Individual-level genotype data linked to real cancer diagnoses is
controlled-access** (UK Biobank, dbGaP, TCGA germline calls, etc.) and
cannot legally or ethically be redistributed in a public GitHub repo. So
rather than pretend otherwise, this project is explicit about exactly
which parts of the data are real and which are simulated:

| Component | Real or simulated? | Source |
| --- | --- | --- |
| GWAS effect weights (76 SNPs) | **Real** | [PGS000055](https://www.pgscatalog.org/score/PGS000055/) (Schmit et al., *J Natl Cancer Inst* 2019), from the PGS Catalog |
| Genotypes (2,504 individuals) | **Real** | 1000 Genomes Project, Phase 3 |
| Ancestry / super-population labels | **Real** | 1000 Genomes sample panel |
| Sex | **Real** | 1000 Genomes sample panel |
| Polygenic risk score itself | **Real** (computed from the two real inputs above, standard weighted-sum formula) | — |
| Age | **Simulated** | Drawn independently, not linked to any real attribute |
| Family history of CRC | **Simulated** | Drawn independently |
| Colorectal cancer case/control label | **Simulated**, via a logistic risk model whose PRS coefficient is calibrated to reproduce the *actually published* AUROC for PGS000055 (0.65, 95% CI 0.62-0.69) | See `scripts/05_simulate_phenotype.py` docstring |

In other words: the genetics are 100% real, and only the disease outcome
needed to be simulated to demonstrate the ML workflow end to end. This is
the same strategy used in PRS methods papers to validate a new approach
before ever touching biobank data.

## What this shows that a toy PRS demo usually doesn't

Most PRS tutorials stop at "multiply genotype by weight, sum, done." Two
things here go a step further and don't require any simulated data at all:

1. **Real allele-frequency-driven population differences.** 1000 Genomes
   spans 5 real ancestry super-populations. Computing the real PRS for all
   of them shows a highly significant difference in mean PRS across
   ancestries (one-way ANOVA F=74.4, p≈1×10⁻⁵⁹) purely from allele
   frequency differences at the scored SNPs — not because any of these
   groups actually has 74-sigma-significant differences in colorectal
   cancer risk. This is the real-world "PRS portability problem": a score
   trained on African American + European GWAS (as PGS000055 was)
   systematically over- or under-estimates risk in a South Asian or East
   Asian genome, and clinical deployment of PRS without ancestry-aware
   recalibration is an active, unsolved problem in the field.
2. **Calibrating the simulation against a real reported effect size**,
   rather than picking an arbitrary one, so the ML results in this repo
   ("PRS-only AUC ≈ 0.66") are directly comparable to the actually
   published number (0.65) for this score.

## Pipeline

| Step | Script | What happens |
| --- | --- | --- |
| 1 | `01_download_pgs_weights.sh` | Downloads PGS000055 (76 SNPs, hg19/GRCh37) from the PGS Catalog |
| 2 | `02_extract_1000genomes_genotypes.sh` | Remote `tabix` queries pull just those 76 positions out of the ~1 GB-per-chromosome 1000 Genomes VCFs — no need to download whole chromosomes for 76 SNPs |
| 3 | `03_compute_prs.py` | Matches effect/other allele to REF/ALT per site (handling multi-allelic positions correctly) and computes `PRS_i = sum(dosage_ij * weight_j)` per person — the same formula used by PLINK `--score` / PRSice |
| 4 | `04_ancestry_transferability_analysis.py` | Real-data-only: PRS distribution across the 5 super-populations, ANOVA |
| 5 | `05_simulate_phenotype.py` | Adds the one simulated layer: case/control label via a calibrated logistic risk model (see data table above) |
| 6 | `06_train_evaluate_models.py` | Logistic regression (PRS-only and PRS+covariates) and gradient boosting, evaluated with 5-fold stratified CV; ROC, calibration, and risk-decile figures |

## Quickstart

```bash
conda env create -f environment.yml
conda activate prs-cancer-ml
bash scripts/run_all.sh
```

Figures land in `figures/`, tables in `results/`. Runs in a couple of
minutes — the "big" 1000 Genomes VCFs are queried remotely for 76 positions
each, never downloaded in full.

## Results

### 1. A Euro/African-American-derived score doesn't transfer evenly across ancestries (100% real data)

![PRS by superpopulation](docs/example_output/07_prs_by_superpopulation.png)

Mean PRS differs significantly across all 5 super-populations (one-way
ANOVA F=74.45, p=1.08×10⁻⁵⁹; full table in
[`docs/example_output/prs_by_superpopulation.csv`](docs/example_output/prs_by_superpopulation.csv)).
Full write-up in
[`docs/example_output/ancestry_anova.txt`](docs/example_output/ancestry_anova.txt).

### 2. ML classifiers on the simulated cohort

| Model | 5-fold CV AUC |
| --- | --- |
| Logistic regression, PRS only | 0.664 ± 0.041 |
| Logistic regression, PRS + age + family history + sex | 0.674 ± 0.050 |
| Gradient boosting, PRS + covariates | 0.641 ± 0.061 |

The PRS-only AUC (0.664) lands right on top of the actually published
AUROC for PGS000055 (0.65) — expected, since that's exactly what the
simulation was calibrated to reproduce. Gradient boosting doesn't beat
plain logistic regression here, which is the expected result when the true
generating model is linear-in-covariates and cases are scarce (~105 of
2,504) — a good reminder that model complexity isn't free.

![ROC curves](docs/example_output/08_roc_curves.png)

Odds ratio per 1-SD increase in PRS: **1.86**. Top-PRS-decile vs.
middle-decile odds of being a case: **3.43×** — both very much in line
with typical published cancer PRS effect sizes.

![Risk decile stratification](docs/example_output/10_risk_decile_stratification.png)

Calibration curve (predicted vs. observed risk by decile) for the best
model:

![Calibration curve](docs/example_output/09_calibration_curve.png)

Full metrics: [`docs/example_output/model_performance.csv`](docs/example_output/model_performance.csv),
[`docs/example_output/prs_decile_odds_ratios.csv`](docs/example_output/prs_decile_odds_ratios.csv).

## Repository structure

```text
.
├── environment.yml
├── scripts/
│   ├── 01_download_pgs_weights.sh
│   ├── 02_extract_1000genomes_genotypes.sh
│   ├── 03_compute_prs.py
│   ├── 04_ancestry_transferability_analysis.py
│   ├── 05_simulate_phenotype.py
│   ├── 06_train_evaluate_models.py
│   └── run_all.sh
├── docs/example_output/          # curated figures/tables shown above
├── .github/workflows/ci.yml      # runs the full pipeline on every push
├── data/, figures/, results/     # created locally at run time (gitignored)
└── LICENSE
```

## Limitations

- **The disease label is simulated.** This is the whole point of the data
  table above, but it's worth repeating: no conclusions about real-world
  colorectal cancer risk should be drawn from the ML section — only the
  *methodology* is meant to generalize.
- **PRS was standardized on the full multi-ancestry cohort**, not within
  each population separately. This mirrors a common (and, per the
  transferability analysis above, clearly imperfect) real-world practice;
  a more rigorous approach would standardize within ancestry or use an
  ancestry-calibrated score.
- **Small case count (~105 of 2,504).** AUC confidence intervals from
  cross-validation are correspondingly wide; a real clinical validation
  would need a much larger, prospectively phenotyped cohort — which is
  exactly why resources like UK Biobank exist and require controlled
  access.
- **Only 76 SNPs.** PGS000055 predates genome-wide PRS methods like
  LDpred/PRS-CS that use millions of variants and generally achieve higher
  AUC; it was chosen here specifically because its small, curated variant
  list keeps genotype extraction fast and auditable for a demo pipeline.

## Skills demonstrated

Polygenic risk score methodology (real GWAS Catalog/PGS Catalog effect
weights, correct effect/other allele matching against VCF REF/ALT,
multi-allelic site handling), large remote-file querying (`tabix` range
queries against multi-gigabyte indexed VCFs without full downloads),
population genetics (1000 Genomes super-populations, PRS ancestry
transferability), simulation methodology calibrated against real published
effect sizes, and applied ML (scikit-learn: logistic regression, gradient
boosting, stratified cross-validation, ROC/AUC, calibration curves, risk
stratification) — plus the judgment to be explicit about what's real data
and what isn't, which matters as much as the code in a domain like this.
