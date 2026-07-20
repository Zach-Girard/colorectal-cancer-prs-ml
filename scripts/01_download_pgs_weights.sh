#!/usr/bin/env bash
#
# 01_download_pgs_weights.sh
#
# Downloads a real, peer-reviewed colorectal cancer polygenic score from the
# PGS Catalog (https://www.pgscatalog.org/score/PGS000055/):
#
#   PGS000055 "PRS_CRC" -- Schmit SL et al., J Natl Cancer Inst (2019).
#   76 genome-wide-significant SNPs, developed/replicated in African
#   American and European ancestry colorectal cancer GWAS.
#
# We use the GRCh37-harmonized version so positions line up directly with
# the 1000 Genomes Phase 3 (GRCh37) VCFs used downstream.
#
# Usage: scripts/01_download_pgs_weights.sh [output_dir]

set -euo pipefail

OUTPUT_DIR="${1:-data/pgs_catalog}"
PGS_ID="PGS000055"
URL="https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/${PGS_ID}/ScoringFiles/Harmonized/${PGS_ID}_hmPOS_GRCh37.txt.gz"
OUT_FILE="${OUTPUT_DIR}/${PGS_ID}_hmPOS_GRCh37.txt.gz"

mkdir -p "${OUTPUT_DIR}"

if [[ -f "${OUT_FILE}" ]]; then
  echo "PGS weight file already present at ${OUT_FILE}, skipping download."
else
  echo "Downloading ${PGS_ID} (Schmit et al. 2019, colorectal cancer, 76 SNPs) from PGS Catalog..."
  curl -sL "${URL}" -o "${OUT_FILE}"
fi

n_variants=$(gzip -dc "${OUT_FILE}" | grep -v '^#' | tail -n +2 | wc -l | tr -d ' ')
echo "Downloaded ${n_variants} variant weights to ${OUT_FILE}"
