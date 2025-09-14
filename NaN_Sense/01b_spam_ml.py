#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
01b_spam_ml.py
---------------
Weakly-supervised ML spam classifier on top of rule-based flags.

Inputs
- --raw:   RAW file (csv/parquet) with comments; we will recompute rule flags here.
- --clean: CLEAN file from 01_ingest_spam_dedup.py (stage1_clean.parquet) used to mine safe "ham".
          If you don't have it yet, you can just pass --raw; we'll sample ham from raw too.

Outputs
- stage1b_scored.parquet      # raw rows + rule flags + spam_prob + is_spam_ml + is_spam_final
- stage1b_clean_ml.parquet    # filtered (not is_spam_final and not exact dup)
- out_spam_ml/metrics.txt     # quick PR metrics on a holdout of the silver set
"""

# To run:
# python 01b_spam_ml.py --raw comments_cleaned.parquet --clean stage1_clean.parquet --out-prefix stage1b --sample 120000 --target-precision 0.95




import os, re, argparse, math, json, hashlib, warnings
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Rule set (same spirit as 01_ingest_spam_dedup.py) ----------
URL_RE = re.compile(r"(https?://|www\.)\S+", re.I)
DOMAIN_RE = re.compile(r"\b[\w-]+\.(com|net|org|io|co|shop|store|xyz|club|info)\b", re.I)
INVITE_RE = re.compile(r"\b(subscribe|follow|promo|discount|sale|giveaway|coupon|code|deal|dm|inbox|whatsapp|wa\.me|telegram|click|buy now|order)\b", re.I)
CONTACT_RE = re.compile(r"(\+\d{6,}|\b\d{3,}[-\s]?\d{3,}[-\s]?\d{3,}\b|@[A-Za-z0-9_.-]+)", re.I)
REPEAT_RE = re.compile(r"(.)\1{3,}")
EMOJI_RE  = re.compile("[" "\U0001F300-\U0001F6FF" "\U0001F900-\U0001FAFF" "\u2600-\u26FF\u2700-\u27BF" "]", flags=re.UNICODE)

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    try:
        import ftfy
        s = ftfy.fix_text(s)
    except Exception:
        pass
    return re.sub(r"\s+", " ", s.replace("\u200b", " ").strip())

def simple_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def spam_rule_flag(txt: str) -> bool:
    t = txt
    lc = t.lower()
    has_url = bool(URL_RE.search(lc) or DOMAIN_RE.search(lc))
    has_inv = bool(INVITE_RE.search(lc) or CONTACT_RE.search(lc))
    many_emoji = len(EMOJI_RE.findall(t)) >= 4
    long_repeat = bool(REPEAT_RE.search(t))
    too_short = len(lc.split()) < 2
    return (has_url and has_inv) or long_repeat or (many_emoji and has_inv) or (has_url and too_short)

# ---------- IO helpers ----------
def load_any(path: str, text_candidates=("clean_text","textOriginal","text","comment")):
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    for c in text_candidates:
        if c in df.columns:
            text_col = c
            break
    else:
        obj = [c for c in df.columns if pd.api.types.is_object_dtype(df[c])]
        if not obj: raise SystemExit("No text-like column found.")
        text_col = obj[0]
    return df, text_col

# ---------- Feature builder ----------
def build_design_matrix(texts: pd.Series, meta: pd.DataFrame):
    """Return sparse X for (char 3-5 + word 1-2) TF-IDF + scaled meta features."""
    # Hashing vectorizers (memory-safe for big corpora), then TF-IDF to weight
    hv_char = HashingVectorizer(analyzer="char", ngram_range=(3,5), n_features=2**20, norm=None, alternate_sign=False) # type: ignore
    hv_word = HashingVectorizer(analyzer="word", ngram_range=(1,2), n_features=2**20, norm=None, alternate_sign=False, lowercase=True, token_pattern=r"(?u)\b[^\W\d_][\w'-]{2,}\b") # pyright: ignore[reportArgumentType]
    Xc = hv_char.transform(texts.fillna(""))
    Xw = hv_word.transform(texts.fillna(""))

    tfidf = TfidfTransformer()
    Xc = tfidf.fit_transform(Xc)
    Xw = tfidf.fit_transform(Xw)

    # numeric meta -> scaled
    num = meta.astype(float).fillna(0.0).to_numpy()
    scaler = StandardScaler(with_mean=False)
    Xn = scaler.fit_transform(num)

    X = hstack([Xc, Xw, Xn], format="csr")
    return X

# ---------- Threshold search ----------
def pick_threshold_for_precision(y_true, y_scores, target_prec=0.95):
    prec, rec, thr = precision_recall_curve(y_true, y_scores)
    best_t = 0.5
    for p, t in zip(prec, np.r_[thr, thr[-1]]):
        if p >= target_prec:
            best_t = t
            break
    ap = average_precision_score(y_true, y_scores)
    return float(best_t), float(ap)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Weakly-supervised ML spam filter (rules → silver labels → LR)")
    ap.add_argument("--raw", required=True, help="RAW comments file (csv or parquet)")
    ap.add_argument("--clean", default=None, help="stage1_clean.parquet from 01 (optional)")
    ap.add_argument("--out-prefix", default="stage1b", help="Output prefix")
    ap.add_argument("--sample", type=int, default=120000, help="Max silver training rows (balanced)")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--target-precision", type=float, default=0.95)
    args = ap.parse_args()

    os.makedirs("out_spam_ml", exist_ok=True)

    # 1) Load RAW and recompute rule flags (positives live here)
    raw, tcol = load_any(args.raw)
    raw = raw.copy()
    raw["__text__"] = raw[tcol].map(normalize_text)
    raw["__dupkey__"] = raw["__text__"].map(simple_hash)
    raw["is_exact_dup"] = raw["__dupkey__"].duplicated(keep="first")
    raw["is_spam_rule"] = raw["__text__"].map(spam_rule_flag)
    raw["len_words"] = raw["__text__"].str.split().map(len)
    raw["url_count"] = raw["__text__"].str.count(URL_RE.pattern)
    raw["emoji_count"] = raw["__text__"].str.count(EMOJI_RE.pattern)

    # 2) Build silver labels
    pos = raw.loc[raw["is_spam_rule"]].copy()
    if args.clean is not None and os.path.exists(args.clean):
        neg_src = pd.read_parquet(args.clean)
        neg_mask = (~neg_src.get("is_spam_rule", False)) & (~neg_src.get("is_exact_dup", False))
        neg = neg_src.loc[neg_mask].copy()
        neg["__text__"] = neg.get("__text__", neg[tcol] if tcol in neg.columns else neg.iloc[:,0]).map(normalize_text) # pyright: ignore[reportCallIssue, reportArgumentType]
        neg["len_words"] = neg["__text__"].str.split().map(len)
        neg["url_count"] = neg["__text__"].str.count(URL_RE.pattern)
        neg["emoji_count"] = neg["__text__"].str.count(EMOJI_RE.pattern)
    else:
        # fall back: mine "likely ham" from RAW
        likely_ham = (~raw["is_spam_rule"]) & (~raw["is_exact_dup"]) & (raw["len_words"].between(5, 80)) & (raw["url_count"]==0)
        neg = raw.loc[likely_ham].copy()

    # Balance & sample
    n_each = min(len(pos), len(neg), args.sample//2 if args.sample else min(len(pos), len(neg)))
    if n_each == 0:
        raise SystemExit("Not enough silver labels to train. Check inputs.")
    pos_s = pos.sample(n_each, random_state=args.random_state)
    neg_s = neg.sample(n_each, random_state=args.random_state)
    silver = pd.concat([pos_s.assign(label=1), neg_s.assign(label=0)], ignore_index=True)
    silver = silver.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)

    # 3) Build features
    meta_cols = ["len_words","url_count","emoji_count"]
    X = build_design_matrix(silver["__text__"], silver[meta_cols])
    y = silver["label"].to_numpy()

    # 4) Train/val split & fit LR (SGD, log_loss)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=args.random_state, stratify=y)
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-5, max_iter=5, n_iter_no_change=2, class_weight="balanced", random_state=args.random_state)
    clf.fit(Xtr, ytr)

    # 5) Tune threshold for precision >= target
    va_scores = clf.predict_proba(Xva)[:,1]
    thr, ap = pick_threshold_for_precision(yva, va_scores, target_prec=args.target_precision)

    # 6) Score ALL RAW rows
    X_all = build_design_matrix(raw["__text__"], raw[meta_cols])
    raw["spam_prob"] = clf.predict_proba(X_all)[:,1].astype("float32") # pyright: ignore[reportArgumentType]
    raw["is_spam_ml"] = (raw["spam_prob"] >= thr)
    raw["is_spam_final"] = raw["is_spam_rule"] | raw["is_spam_ml"]

    # 7) Save scored + cleaned
    scored_path = f"{args.out_prefix}_scored.parquet"
    raw.to_parquet(scored_path, index=False)

    clean_ml = raw.loc[~raw["is_spam_final"] & ~raw["is_exact_dup"]].copy()
    clean_path = f"{args.out_prefix}_clean_ml.parquet"
    clean_ml.to_parquet(clean_path, index=False)

    # 8) Write quick metrics
    pr, rc, thr_grid = precision_recall_curve(yva, va_scores)
    report = classification_report(yva, (va_scores >= thr).astype(int), digits=3)
    with open("out_spam_ml/metrics.txt","w",encoding="utf-8") as f:
        f.write(json.dumps({
            "silver_pos": int((silver['label']==1).sum()),
            "silver_neg": int((silver['label']==0).sum()),
            "avg_precision_val": float(ap),
            "chosen_threshold": float(thr),
            "target_precision": float(args.target_precision)
        }, indent=2))
        f.write("\n\nclassification_report@\n")
        f.write(report) # pyright: ignore[reportArgumentType]

    print(f"[spam-ml] trained on silver set n={len(silver):,} (pos={int((silver['label']==1).sum()):,}, neg={int((silver['label']==0).sum()):,})")
    print(f"[spam-ml] AP={ap:.3f}, threshold={thr:.3f} for precision≥{args.target_precision:.2f}")
    print(f"[spam-ml] wrote: {scored_path} (scored all rows), {clean_path} (filtered rows)")
    print(f"[spam-ml] metrics: out_spam_ml/metrics.txt")
    
if __name__ == "__main__":
    main()
