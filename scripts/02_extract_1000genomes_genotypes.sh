#!/usr/bin/env bash
#
# 02_extract_1000genomes_genotypes.sh
#
# Pulls real genotypes for the 76 PGS000055 SNPs, for all 2,504 individuals
# in the 1000 Genomes Project Phase 3 release, directly from EBI's remote,
# tabix-indexed VCFs -- no need to download whole per-chromosome VCFs
# (each ~1 GB) to get 76 positions out of them.
#
# Also downloads the accompanying sample -> population / super-population
# panel, which is what makes the ancestry-transferability analysis in
# scripts/04 possible.
#
# Usage: scripts/02_extract_1000genomes_genotypes.sh [pgs_weights_file] [output_dir]

set -euo pipefail

PGS_WEIGHTS="${1:-data/pgs_catalog/PGS000055_hmPOS_GRCh37.txt.gz}"
OUTPUT_DIR="${2:-data/1000genomes}"
BASE_URL="https://ftp.ebi.ac.uk/1000g/ftp/release/20130502"
PANEL_URL="${BASE_URL}/integrated_call_samples_v3.20130502.ALL.panel"

mkdir -p "${OUTPUT_DIR}"

echo "Downloading sample population panel..."
curl -sL "${PANEL_URL}" -o "${OUTPUT_DIR}/sample_populations.panel"
echo "  $(($(wc -l < "${OUTPUT_DIR}/sample_populations.panel") - 1)) samples across 5 super-populations"

# chrom -> "pos1,pos2,..." (skip header/comment lines from the PGS file)
echo "Building per-chromosome SNP position lists from ${PGS_WEIGHTS}..."
chroms=$(gzip -dc "${PGS_WEIGHTS}" | grep -v '^#' | tail -n +2 | cut -f2 | sort -n -u)

VCF_SLICE="${OUTPUT_DIR}/pgs000055_snps.vcf"
: > "${VCF_SLICE}"
header_written=false

for chrom in ${chroms}; do
  vcf_url="${BASE_URL}/ALL.chr${chrom}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"

  # Build "chr:pos-pos" region args for every SNP on this chromosome.
  regions=$(gzip -dc "${PGS_WEIGHTS}" | grep -v '^#' | tail -n +2 | \
    awk -F'\t' -v c="${chrom}" '$2==c {print $2":"$3"-"$3}')

  echo "  chr${chrom}: querying $(echo "${regions}" | wc -w | tr -d ' ') position(s)..."

  if [[ "${header_written}" == false ]]; then
    tabix -h "${vcf_url}" ${regions} >> "${VCF_SLICE}"
    header_written=true
  else
    tabix "${vcf_url}" ${regions} >> "${VCF_SLICE}"
  fi
done

n_variants_found=$(grep -vc '^#' "${VCF_SLICE}")
n_variants_total=$(gzip -dc "${PGS_WEIGHTS}" | grep -v '^#' | tail -n +2 | wc -l | tr -d ' ')
echo "Extracted ${n_variants_found} VCF records / ${n_variants_total} scored variant positions to ${VCF_SLICE}"
# It's normal for the VCF record count to come out slightly *higher* than
# the scored variant count: a handful of these positions are multi-allelic
# in 1000 Genomes (more than one alternate allele observed there), so tabix
# returns more than one record for that position. scripts/03_compute_prs.py
# is the step that actually figures out which record (if any) matches the
# PGS SNP's effect/other allele.
