#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pack key CSV outputs into a `results/` folder for committing/sharing with judges.

Copies (if present):
  - out_qer/qer_summary.csv
  - out_categories/category_breakdown_by_pillar.csv
  - out_pain/pain_counts_by_pillar.csv
  - out_pain/pain_counts_by_pillar_period.csv
  - out_trends/top_risers_by_pillar.csv
  - out_trends/term_trendlines.csv
"""
import os, shutil
from pathlib import Path

MAP = [
    ("out_qer/qer_summary.csv", "results/qer_summary.csv"),
    ("out_categories/category_breakdown_by_pillar.csv", "results/category_breakdown_by_pillar.csv"),
    ("out_pain/pain_counts_by_pillar.csv", "results/pain_counts_by_pillar.csv"),
    ("out_pain/pain_counts_by_pillar_period.csv", "results/pain_counts_by_pillar_period.csv"),
    ("out_trends/top_risers_by_pillar.csv", "results/top_risers_by_pillar.csv"),
    ("out_trends/term_trendlines.csv", "results/term_trendlines.csv"),
]

def main():
    dest_dir = Path("results")
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src, dst in MAP:
        s = Path(src)
        if s.exists():
            d = Path(dst)
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
            print(f"[pack] copied {s} -> {d}")
            copied += 1
        else:
            print(f"[pack] missing: {s} (skipped)")
    if copied == 0:
        print("[pack] No files were copied. Did you run the pipeline?")
    else:
        print(f"[pack] Done. Files in {dest_dir.resolve()}")

if __name__ == "__main__":
    main()
