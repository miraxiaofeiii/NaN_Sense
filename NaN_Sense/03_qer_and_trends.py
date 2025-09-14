
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# To run:
# python 03_qer_and_trends.py --input stage2_tagged.parquet --period W --min-df 50

import argparse, os, re, json
import pandas as pd
import numpy as np

EMO_W = {"joy": 0.2, "neutral": 0.0, "sadness": -0.1, "anger": -0.2}

# ---------------- Pain point lexicon (REVISED) ----------------
# Design:
# - use word boundaries and explicit phrases
# - currency tokens must include a number (e.g., RM59, $12); plain "$" or "rm" won't match
# - keep short generic words OUT (e.g., "cost") unless paired with sentiment in tag_pain_points()

PAIN_PRIORITY = ["Price","Packaging","Quality","Availability","Sustainability","Compatibility"]

PAIN_LEXICON = {
    # Split price into "sentiment cues" and "currency mentions".
    # We'll only keep Price if either a sentiment cue is present OR a currency + price cue co-occur.
    "Price_SENTIMENT": [
        r"\boverpriced\b",
        r"\btoo\s+expensive\b",
        r"\bexpensive\b",
        r"\bpricey\b",
        r"\bcheap\b",
        r"\baffordable\b",
        r"\bvalue\s+for\s+money\b",
        r"\bworth\s+the\s+price\b",
    ],
    "Price_CURRENCY": [
        r"\b(?:rm|myr)\s?\d{1,6}\b",          # RM59, myr 120
        r"(?:^|\s)\$\s?\d{1,6}\b",            # $12
        r"\b(?:usd|sgd|idr|inr|eur)\s?\d{1,6}\b",
    ],

    "Packaging": [
        r"\bpackag(?:e|ing)\b", r"\bpump\b", r"\bnozzle\b", r"\bapplicator\b",
        r"\bcap\b", r"\bbottle\b", r"\btube\b", r"\bbox\b", r"\bseal\b", r"\blid\b",
        r"\bleak(?:ing|s)?\b", r"\bspill(?:ed|s|ing)?\b", r"\bbroken\b"
    ],

    "Quality": [
        r"\bqualit(?:y|ies)\b", r"\bpigment(?:ation)?\b", r"\bcoverage\b", r"\blongev(?:ity)?\b",
        r"\blast(?:ing)?\b", r"\bblend(?:s|ed|ing)?\b", r"\btexture\b",
        r"\bstick(?:y)?\b", r"\bgreas(?:e|y)\b", r"\bdry(?:ing)?\b", r"\bpatch(?:y|iness)\b",
        r"\bcak(?:e|ey|ing)\b", r"\bflake(?:s|y|ing)?\b", r"\bsmudge(?:s|d|ing)?\b",
        r"\btransfer(?:s|red|ing)?\b", r"\boxidiz(?:e|es|ed|ing)\b"
    ],

    "Availability": [
        r"\bout\s+of\s+stock\b", r"\bsold\s+out\b", r"\brestock(?:ed|ing)?\b",
        r"\bwhere\s+to\s+buy\b", r"\bnot\s+available\b", r"\bcannot\s+find\b",
        r"\bavailability\b", r"\bstore\b", r"\bonline\s+only\b"
    ],

    "Sustainability": [
        r"\bsustainab(?:le|ility)\b", r"\brecycl(?:e|ing|able)\b",
        r"\brefill(?:able)?\b", r"\beco\b", r"\benviron(?:ment|mental)\b",
        r"\bwaste\b", r"\bplastic\b", r"\bcarbon\b", r"\bcruelty\s*free\b", r"\bvegan\b",
        r"\bpalm\s+oil\b", r"\bgreen\b"
    ],

    "Compatibility": [
        r"\bsensitive\b", r"\ballerg(?:y|ic)\b", r"\brash\b", r"\bbreakout(?:s)?\b",
        r"\bacne\b", r"\boily\b", r"\bdry\s*skin\b", r"\bcombination\s*skin\b",
        r"\bmatch\b", r"\bshade\b", r"\btone\b", r"\bundertone\b", r"\boxidiz(?:e|es|ed|ing)\b",
        r"\bfits?\s*me\b"
    ],
}

import re as _re

def _compile_pain_patterns():
    rx = {}
    for k, patterns in PAIN_LEXICON.items():
        rx[k] = _re.compile("|".join(patterns), _re.IGNORECASE)
    return rx

PAIN_RX = _compile_pain_patterns()

def tag_pain_points(text: str):
    """
    Rules:
    - 'Price' requires either a price SENTIMENT cue, OR (currency mention AND any price cue word nearby).
    - Currency-only mentions do NOT count as Price.
    - Everything uses word boundaries to avoid 'rm' hitting 'cream', etc.
    """
    t = (text or "").strip()
    if not t:
        return []

    hits = set()

    # 2.a) Non-price buckets (straightforward)
    for label in ["Packaging","Quality","Availability","Sustainability","Compatibility"]:
        if PAIN_RX[label].search(t):
            hits.add(label)

    # 2.b) Price logic
    price_sent = bool(PAIN_RX["Price_SENTIMENT"].search(t))
    price_curr = bool(PAIN_RX["Price_CURRENCY"].search(t))

    # Accept if we have explicit sentiment OR currency + any price sentiment cue word present
    # (price_sent already encodes cue words like "expensive", "cheap", etc.)
    if price_sent or (price_curr and price_sent):
        hits.add("Price")

    # Optional: if nothing else matched but "cost" appears with a sentiment word nearby (±3 tokens), count as Price
    # (kept conservative — NOT using plain "cost" globally)
    # You can uncomment if you later want this behavior.

    return sorted(hits)

def choose_primary(labels):
    for k in PAIN_PRIORITY:
        if k in labels: return k
    return labels[0] if not labels else None

def choose_primary(labels):
    for k in PAIN_PRIORITY:
        if k in labels:
            return k
    return labels[0] if labels else None

def add_pain(df: pd.DataFrame, text_col="__text__") -> pd.DataFrame:
    dff = df.copy()

    # pillar-aware gating (lightweight):
    # if a row matches no pillar flags, we still allow pain tagging (generic comments).
    # otherwise, we run the same detector. (You can make this stricter later by mapping
    # certain pains to certain pillars only.)
    pains = dff[text_col].astype(str).map(tag_pain_points)

    # has_pain now requires at least ONE real label (currency-only no longer passes tag_pain_points)
    dff["pain_points"] = pains
    dff["has_pain"] = pains.map(lambda xs: len(xs) > 0)
    dff["pain_primary"] = pains.map(choose_primary)
    return dff

def length_bucket(s: str):
    n = len((s or "").split())
    return 0.3 if n < 20 else (0.7 if n <= 80 else 1.0)

def compute_qer(df: pd.DataFrame, likes_col="likeCount"):
    if likes_col not in df.columns: df[likes_col] = 0
    lw = df["__text__"].map(length_bucket).astype(float)
    ew = df["emotion_base"].str.lower().map(EMO_W).fillna(0.0)
    likes = pd.to_numeric(df[likes_col], errors="coerce").fillna(0)
    return (lw + ew) * np.log1p(likes)

def zburst_tf(counts):
    totals = counts.sum(axis=1, keepdims=True)
    tf = counts / np.maximum(totals, 1)
    T = pd.DataFrame(tf)
    base_mean = T.rolling(3, min_periods=1).mean().shift(1).to_numpy()
    base_std  = T.rolling(3, min_periods=1).std(ddof=1).shift(1).fillna(0.0).to_numpy()
    eps = 1e-9
    return (tf - base_mean) / (base_std + eps)

def tokenize(texts: pd.Series):
    # very light normalisation, unigrams+bigrams
    import sklearn.feature_extraction.text as sktxt
    TOKEN = r"(?u)\b[^\W\d_][\w'-]{2,}\b"
    vec = sktxt.CountVectorizer(lowercase=True, token_pattern=TOKEN, ngram_range=(1,2), min_df=1, stop_words="english")
    X = vec.fit_transform(texts.fillna(""))
    vocab = np.array(vec.get_feature_names_out())
    # remove domain-generic words
    EXTRA = {"skin","hair","face","product","smell","fragrance","perfume","makeup","skincare"}
    keep = ~np.isin(vocab, list(EXTRA))
    return X[:, keep], vocab[keep] # pyright: ignore[reportIndexIssue]

def trends(df: pd.DataFrame, period="W", time_col="publishedAt", text_col="__text__",
           min_df=50, emotion=None, pillars=("skincare","makeup","haircare","fragrance")):
    out_dir = "out_trends"; os.makedirs(out_dir, exist_ok=True)
    if emotion:
        df = df[df["emotion_base"].str.lower().eq(emotion.lower())]
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    df["__period"] = df[time_col].dt.to_period(period).dt.to_timestamp()

    # vectorize once on full corpus for stable vocab
    X, vocab = tokenize(df[text_col])
    df = df.reset_index(drop=True)

    # filter rare terms
    docfreq = np.asarray((X > 0).sum(axis=0)).ravel()
    keep = docfreq >= min_df
    X = X[:, keep]; vocab = vocab[keep]

    rows = []; lines = []
    for p in pillars:
        col = f"is_{p}"
        if col not in df.columns:  # allow overall trends if pillars missing
            mask = np.ones(len(df), dtype=bool)
        else:
            mask = df[col].astype(bool).values
            if mask.sum() == 0: continue # pyright: ignore[reportAttributeAccessIssue]

        Xp = X[mask]; per = df.loc[mask, "__period"] # pyright: ignore[reportArgumentType, reportCallIssue]
        periods = np.sort(per.unique())
        # counts per period
        cps = []
        for t in periods:
            sel = (per == t).values
            cps.append(np.asarray(Xp[sel].sum(axis=0)).ravel() if sel.sum() else np.zeros(Xp.shape[1]))
        counts = np.vstack(cps)
        zb = zburst_tf(counts)

        # top risers per period
        for i, t in enumerate(periods):
            order = np.argsort(zb[i])[::-1][:30]
            for j in order:
                rows.append({
                    "pillar": p,
                    "period": t,
                    "term": vocab[j],
                    "trend_score": float(zb[i, j]),
                    "count": int(counts[i, j])
                })

        # trendlines for overall top terms
        overall = zb.max(axis=0)
        top_idx = np.argsort(overall)[::-1][:min(50, len(overall))]
        for j in top_idx:
            for i, t in enumerate(periods):
                tot = counts[i].sum() or 1
                lines.append({
                    "pillar": p, "term": vocab[j], "period": t,
                    "share": float(counts[i, j] / tot),
                    "count": int(counts[i, j]),
                    "trend_score": float(zb[i, j]),
                })

    pd.DataFrame(rows).sort_values(["pillar","period","trend_score"], ascending=[True,True,False])         .to_csv(os.path.join(out_dir, "top_risers_by_pillar.csv"), index=False)
    pd.DataFrame(lines).to_csv(os.path.join(out_dir, "term_trendlines.csv"), index=False)
    print("[trends] saved out_trends/top_risers_by_pillar.csv & term_trendlines.csv")

# ---------------- NEW: category breakdown + pain exports ----------------
def export_category_breakdown(df: pd.DataFrame):
    if "category_primary" not in df.columns:
        print("[warn] category_primary not found (run 02 script with subcategory tagging). Skipping category breakdown export.")
        return
    os.makedirs("out_categories", exist_ok=True)
    records = []
    for p in ["skincare","makeup","haircare","fragrance"]:
        col = f"is_{p}"
        if col not in df.columns: continue
        g = df[df[col]].groupby("category_primary").size().reset_index(name="count")
        for _, r in g.iterrows():
            records.append({"pillar": p, "category": r["category_primary"], "count": int(r["count"])})
    pd.DataFrame(records).to_csv("out_categories/category_breakdown_by_pillar.csv", index=False)
    print("[cats] saved out_categories/category_breakdown_by_pillar.csv")

def export_pain_point_tables(df: pd.DataFrame, period="W", time_col=None):
    os.makedirs("out_pain", exist_ok=True)
    # by pillar
    rows = []
    for p in ["skincare","makeup","haircare","fragrance"]:
        col = f"is_{p}"
        if col not in df.columns: continue
        d = df[df[col] & df.get("has_pain", False)]
        counts = (d["pain_primary"].value_counts() if "pain_primary" in d.columns else pd.Series(dtype=int))
        for k in PAIN_PRIORITY:
            rows.append({"pillar": p, "pain_point": k, "count": int(counts.get(k, 0))})
    pd.DataFrame(rows).to_csv("out_pain/pain_counts_by_pillar.csv", index=False)

    # by pillar + period
    if time_col is None:
        time_col = "publishedAt" if "publishedAt" in df.columns else "__period"
    dff = df.copy()
    dff[time_col] = pd.to_datetime(dff[time_col], errors="coerce")
    dff = dff.dropna(subset=[time_col])
    dff["__period"] = dff[time_col].dt.to_period(period).dt.to_timestamp()
    recs = []
    for p in ["skincare","makeup","haircare","fragrance"]:
        col = f"is_{p}"
        if col not in dff.columns: continue
        D = dff[dff[col] & dff.get("has_pain", False)]
        for t, DD in D.groupby("__period"):
            counts = DD["pain_primary"].value_counts() if "pain_primary" in DD.columns else pd.Series(dtype=int)
            for k in PAIN_PRIORITY:
                recs.append({"pillar": p, "period": t, "pain_point": k, "count": int(counts.get(k, 0))})
    pd.DataFrame(recs).to_csv("out_pain/pain_counts_by_pillar_period.csv", index=False)
    print("[pain] saved out_pain/pain_counts_by_pillar.csv & pain_counts_by_pillar_period.csv")

def main():
    ap = argparse.ArgumentParser(description="QER + Trends + Pain points + Category breakdown")
    ap.add_argument("--input", default="stage2_tagged.parquet")
    ap.add_argument("--likes-col", default="likeCount")
    ap.add_argument("--period", choices=["W","M"], default="W")
    ap.add_argument("--min-df", type=int, default=50)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)

    # NEW: pain tagging (multi-label + primary)
    df = add_pain(df, text_col="__text__")

    # QER
    df["qer_score"] = compute_qer(df, likes_col=args.likes_col)

    os.makedirs("out_qer", exist_ok=True)
    def agg_by(col):
        if col not in df.columns: return None
        g = df[df[col]].copy()
        if len(g)==0: return None
        return {"n": len(g), "qer_mean": g["qer_score"].mean()}
    rows = [{"scope":"ALL","n":len(df),"qer_mean":df["qer_score"].mean()}]
    for p in ["is_skincare","is_makeup","is_haircare","is_fragrance"]:
        m = agg_by(p)
        if m: rows.append({"scope": p.replace("is_",""), **m})
    pd.DataFrame(rows).to_csv("out_qer/qer_summary.csv", index=False)
    print("[qer] saved out_qer/qer_summary.csv")

    # Exports for dashboards
    export_category_breakdown(df)
    export_pain_point_tables(df, period=args.period, time_col=("publishedAt" if "publishedAt" in df.columns else "__period"))

    # Trends overall weekly/monthly + anger-only trends
    trends(df, period=args.period, time_col=("publishedAt" if "publishedAt" in df.columns else "__period"),
           text_col="__text__", min_df=args.min_df, emotion=None)
    trends(df, period="M", time_col=("publishedAt" if "publishedAt" in df.columns else "__period"),
           text_col="__text__", min_df=max(30, args.min_df//2), emotion="anger")

if __name__ == "__main__":
    main()
