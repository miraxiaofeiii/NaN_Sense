import pandas as pd
import matplotlib.pyplot as plt

# === Load your inputs ===
# QER summary from stage 3
qer = pd.read_csv("out_qer/qer_summary.csv")

# Stage 2 tagged dataset (to count SoE = comments + likes per pillar)
df = pd.read_parquet("stage2_tagged.parquet")

pillars = ["skincare","makeup","haircare","fragrance"]

# === Compute SoE (likes + comments) ===
soe_data = []
for p in pillars:
    mask = df[f"is_{p}"]
    n_comments = mask.sum()
    likes = df.loc[mask, "likeCount"].sum()
    soe_data.append({"pillar": p, "engagements": n_comments + likes})

soe = pd.DataFrame(soe_data)
soe["soe_share"] = soe["engagements"] / soe["engagements"].sum()

# === Merge with QER ===
qer = qer.rename(columns={"scope":"pillar"})
merged = pd.merge(soe, qer[["pillar","qer_mean"]], on="pillar", how="inner")

# === Plot ===
plt.figure(figsize=(8,6))
x = merged["soe_share"] * 100   # % scale
y = merged["qer_mean"]

# quadrant thresholds = means (or medians)
x_mid = x.mean()
y_mid = y.mean()

plt.axvline(x_mid, color="grey", linestyle="--", lw=1)
plt.axhline(y_mid, color="grey", linestyle="--", lw=1)

for i, row in merged.iterrows():
    plt.scatter(row["soe_share"]*100, row["qer_mean"], s=200, label=row["pillar"].capitalize())
    plt.text(row["soe_share"]*100+0.2, row["qer_mean"]+0.001, row["pillar"].capitalize(), fontsize=10)

plt.title("SoE vs QER by Pillar")
plt.xlabel("Share of Engagement (%)")
plt.ylabel("QER Index (avg per comment)")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("soe_qer_matrix.png", dpi=300)
plt.show()
