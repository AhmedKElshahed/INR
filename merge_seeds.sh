#!/usr/bin/env bash
# Merge per-seed Kaggle downloads into one results_3d_seeds.csv, then aggregate.
# Usage: bash merge_seeds.sh seed0.csv seed1.csv seed2.csv
set -e
out=results_3d_seeds.csv
head -1 "$1" > "$out"                 # header once
for f in "$@"; do tail -n +2 "$f" >> "$out"; done   # data rows from each
echo "merged $# files -> $out ($(($(wc -l < "$out") - 1)) rows)"
python aggregate_seeds.py
