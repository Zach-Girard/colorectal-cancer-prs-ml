#!/usr/bin/env python3
"""
03_compute_prs.py

Computes a real per-individual polygenic risk score (PRS) for colorectal
cancer, using:
  - real published GWAS effect weights (PGS000055, Schmit et al. 2019,
    76 SNPs, from the PGS Catalog)
  - real genotypes for those SNPs in all 2,504 individuals from the 1000
    Genomes Project Phase 3 release

For each scored SNP, the dosage of the *effect* allele (0, 1, or 2 copies)
is read directly from each sample's genotype, then:

    PRS_i = sum over SNPs j of (dosage_ij * effect_weight_j)

which is exactly what tools like PLINK --score or PRSice compute -- there's
no shortcut or approximation here, just the standard weighted-sum formula
applied to real data.

The part of this that wasn't obvious to me at first: a GWAS paper's
"effect allele" for a SNP is just whichever allele the original study
happened to model risk relative to -- it is not guaranteed to be the ALT
allele in someone else's VCF. Silently assuming effect_allele == ALT would
silently flip the sign of that SNP's contribution for any site where it
isn't, which would bias the whole score without throwing an error.
match_dosage() below checks REF/ALT against effect/other allele explicitly
and flips the dosage (2 - ALT_count) when the effect allele turns out to
be REF instead, and reports/skips anything that doesn't match either way
so that's visible rather than silent.

Usage:
    python scripts/03_compute_prs.py \
        --weights data/pgs_catalog/PGS000055_hmPOS_GRCh37.txt.gz \
        --vcf data/1000genomes/pgs000055_snps.vcf \
        --panel data/1000genomes/sample_populations.panel \
        --output results/prs_scores.csv
"""
import argparse
import gzip
import os
import sys

import numpy as np
import pandas as pd


def load_pgs_weights(path: str) -> pd.DataFrame:
    """Load a PGS Catalog scoring file; strip '#' metadata; return SNP weight table.

    Returns columns: rsID, chrom, pos, effect_allele, other_allele, effect_weight.
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        lines = [l for l in fh if not l.startswith("#")]
    from io import StringIO
    df = pd.read_csv(StringIO("".join(lines)), sep="\t")
    df = df.rename(columns={"chr_name": "chrom", "chr_position": "pos"})
    df["chrom"] = df["chrom"].astype(str)
    df["pos"] = df["pos"].astype(int)
    return df[["rsID", "chrom", "pos", "effect_allele", "other_allele", "effect_weight"]]


def load_vcf_genotypes(path: str):
    """Parse the scripts/02 VCF slice into ALT dosages per sample.

    Skips multi-allelic rows (comma in ALT). Converts genotypes (0/1 or 0|1)
    to ALT-allele dosage 0/1/2.

    Returns:
        records: list of {chrom, pos, ref, alt, dosage} dicts
        sample_ids: sample IDs from the VCF header, aligned with dosage arrays
    """
    records = []
    sample_ids = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            fields = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                sample_ids = fields[9:]
                continue
            chrom, pos, _id, ref, alt = fields[0], int(fields[1]), fields[2], fields[3], fields[4]
            if "," in alt:
                # Multi-allelic; PGS000055 is biallelic-only — skip this row.
                continue
            genotypes = fields[9:]
            dosage = np.array(
                [gt.replace("|", "/").split("/").count("1") for gt in genotypes],
                dtype=np.int8,
            )
            records.append({"chrom": chrom, "pos": pos, "ref": ref, "alt": alt, "dosage": dosage})
    return records, sample_ids


def match_dosage(record, effect_allele, other_allele):
    """Map VCF ALT dosage onto the PGS effect-allele dosage for one SNP.

    Returns the 0/1/2 effect-allele dosage array, or None if this VCF row's
    alleles do not match the scoring-file alleles (e.g. wrong ALT at a
    multi-allelic site).
    """
    # Effect allele == ALT: use stored ALT dosage as-is.
    if record["ref"] == other_allele and record["alt"] == effect_allele:
        return record["dosage"]
    # Effect allele == REF: flip dosage (diploid: REF count = 2 - ALT count).
    if record["ref"] == effect_allele and record["alt"] == other_allele:
        return 2 - record["dosage"]
    return None


def compute_prs(weights: pd.DataFrame, vcf_records, sample_ids):
    """Compute PRS_i = sum_j (effect_allele_dosage_ij * weight_j) for all samples.

    For each scored SNP, finds a matching VCF row at (chrom, pos), resolves
    effect-allele dosage via match_dosage(), and accumulates the weighted sum.
    Unmatched SNPs are skipped and returned in the dropped_* lists.

    Returns:
        prs: raw PRS array (length = n samples)
        used: rsIDs included in the score
        dropped_no_record: rsIDs with no genotype at that position
        dropped_allele_mismatch: rsIDs with position hit but allele mismatch
    """
    n_samples = len(sample_ids)
    prs = np.zeros(n_samples, dtype=float)
    used, dropped_no_record, dropped_allele_mismatch = [], [], []

    # Index by (chrom, pos); multi-allelic sites may have multiple records.
    by_pos = {}
    for rec in vcf_records:
        by_pos.setdefault((rec["chrom"], rec["pos"]), []).append(rec)

    for _, snp in weights.iterrows():
        key = (snp["chrom"], snp["pos"])
        candidates = by_pos.get(key, [])
        if not candidates:
            dropped_no_record.append(snp["rsID"])
            continue

        dosage = None
        for rec in candidates:
            dosage = match_dosage(rec, snp["effect_allele"], snp["other_allele"])
            if dosage is not None:
                break

        if dosage is None:
            dropped_allele_mismatch.append(snp["rsID"])
            continue

        prs += dosage * snp["effect_weight"]
        used.append(snp["rsID"])

    return prs, used, dropped_no_record, dropped_allele_mismatch


def main():
    """Load weights + genotypes, compute PRS, merge ancestry panel, write CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="PGS Catalog harmonized scoring file")
    parser.add_argument("--vcf", required=True, help="Extracted 1000 Genomes VCF slice")
    parser.add_argument("--panel", required=True, help="1000 Genomes sample -> population panel")
    parser.add_argument("--output", required=True, help="Output CSV of per-sample PRS")
    args = parser.parse_args()

    weights = load_pgs_weights(args.weights)
    print(f"Loaded {len(weights)} SNP weights from {args.weights}", file=sys.stderr)

    vcf_records, sample_ids = load_vcf_genotypes(args.vcf)
    print(f"Loaded genotypes for {len(vcf_records)} VCF records x {len(sample_ids)} samples", file=sys.stderr)

    prs, used, dropped_no_record, dropped_allele_mismatch = compute_prs(weights, vcf_records, sample_ids)
    print(
        f"PRS computed from {len(used)}/{len(weights)} scored SNPs "
        f"({len(dropped_no_record)} had no matching genotype record, "
        f"{len(dropped_allele_mismatch)} had an allele mismatch at that position)",
        file=sys.stderr,
    )
    if dropped_no_record:
        print(f"  No record found: {dropped_no_record}", file=sys.stderr)
    if dropped_allele_mismatch:
        print(f"  Allele mismatch: {dropped_allele_mismatch}", file=sys.stderr)

    panel = pd.read_csv(args.panel, sep="\t")
    panel.columns = [c.strip() for c in panel.columns]
    panel = panel.rename(columns={"sample": "sample_id"})

    out = pd.DataFrame({"sample_id": sample_ids, "prs_raw": prs})
    # Cohort-standardized PRS (mean 0, SD 1) for downstream models.
    out["prs_z"] = (out["prs_raw"] - out["prs_raw"].mean()) / out["prs_raw"].std()
    out["n_variants_used"] = len(used)
    out = out.merge(panel[["sample_id", "pop", "super_pop", "gender"]], on="sample_id", how="left")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} per-sample PRS values to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
