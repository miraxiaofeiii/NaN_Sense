# To quickly check the results of the pain points detection 

import pandas as pd
from pathlib import Path

# --- File paths ---
pain_by_pillar = Path("out_pain/pain_counts_by_pillar.csv")
pain_by_period = Path("out_pain/pain_counts_by_pillar_period.csv")
cat_breakdown   = Path("out_categories/category_breakdown_by_pillar.csv")

# --- Pain point counts by pillar ---
if pain_by_pillar.exists():
    df_pain = pd.read_csv(pain_by_pillar)
    print("\n=== Consumer Pain Points by Pillar ===")
    print(df_pain.groupby("pillar")["count"].sum())
    print("\nTop pain points overall:")
    print(df_pain.groupby("pain_point")["count"].sum().sort_values(ascending=False).head(10))
else:
    print(f"[WARN] {pain_by_pillar} not found. Did you run the 03 script?")

# --- Pain points by pillar + period ---
if pain_by_period.exists():
    df_pp = pd.read_csv(pain_by_period)
    print("\n=== Pain Points by Pillar and Period (sample) ===")
    print(df_pp.head(20))
else:
    print(f"[WARN] {pain_by_period} not found.")

# --- Category breakdown by pillar ---
if cat_breakdown.exists():
    df_cat = pd.read_csv(cat_breakdown)
    print("\n=== Category Breakdown by Pillar ===")
    print(df_cat.groupby("pillar")["count"].sum())
    print("\nTop categories overall:")
    print(df_cat.groupby("category")["count"].sum().sort_values(ascending=False).head(10))
else:
    print(f"[WARN] {cat_breakdown} not found.")


# Coverage = fraction of comments with any detected pain point
total_by_pillar = df_cat.groupby("pillar")["count"].sum()  # from category CSV
pain_by_pillar  = df_pain.groupby("pillar")["count"].sum()
coverage = (pain_by_pillar / total_by_pillar).fillna(0)
print("\nPain-point coverage by pillar (% of comments with a detected pain):")
print((coverage * 100).round(2))

# ---- Top-3 pain points per pillar (share within pillar) ----
if 'df_pain' in globals() and df_pain is not None:
    gp = df_pain.groupby(["pillar", "pain_point"], as_index=False)["count"].sum()
    # share within each pillar
    gp["share"] = gp["count"] / gp.groupby("pillar")["count"].transform("sum").replace(0, 1)
    top3_pain = (gp.sort_values(["pillar", "share"], ascending=[True, False])
                   .groupby("pillar", group_keys=False)
                   .head(3))
    print("\nTop-3 pain points per pillar (share of painful comments within pillar):")
    for p, rows in top3_pain.groupby("pillar"):
        print(f"\n{p}:")
        for _, r in rows.iterrows():
            print(f"  {r['pain_point']}: {r['share']*100:.2f}%")
else:
    print("\n[INFO] No pain-point rows to summarise.")


# ---- Top-3 categories per pillar (share of all comments in pillar) ----
if 'df_cat' in globals() and df_cat is not None:
    gc = df_cat.groupby(["pillar", "category"], as_index=False)["count"].sum()
    gc["share"] = gc["count"] / gc.groupby("pillar")["count"].transform("sum").replace(0, 1)
    top3_cat = (gc.sort_values(["pillar", "share"], ascending=[True, False])
                  .groupby("pillar", group_keys=False)
                  .head(3))
    print("\nTop-3 categories per pillar (share of all comments in pillar):")
    for p, rows in top3_cat.groupby("pillar"):
        print(f"\n{p}:")
        for _, r in rows.iterrows():
            print(f"  {r['category']}: {r['share']*100:.2f}%")
else:
    print("\n[INFO] No category rows to summarise.")

