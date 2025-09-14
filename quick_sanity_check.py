import pandas as pd
df = pd.read_parquet("stage2_tagged.parquet")
print(df.shape)
print(df.columns.tolist())

pillar_cols = ["is_skincare","is_makeup","is_haircare","is_fragrance","is_loreal_brands"]
print(df[pillar_cols].sum().sort_values(ascending=False))

print(df["emotion_base"].value_counts(normalize=True).round(3))

print(df[["__text__","is_skincare","is_makeup","is_haircare","is_fragrance","emotion_base","emotion_conf"]].sample(10))

print(df[["emotion_base","p_joy","p_neutral","p_sadness","p_anger"]].sample(5))

for cat in ["skincare","makeup","haircare","fragrance","loreal_brands"]:
    print(f"\n--- {cat.upper()} EXAMPLES ---")
    print(df.loc[df[f"is_{cat}"], ["__text__"]].sample(5, random_state=1))

qer = pd.read_csv("out_qer/qer_summary.csv")
print(qer.round(3).to_string(index=False))

lines = pd.read_csv("out_trends/term_trendlines.csv")
print(lines[lines["term"].isin(["oxidise","shade","greasy"])].head(20))

