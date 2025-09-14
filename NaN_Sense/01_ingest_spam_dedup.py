#!/usr/bin/env python
# -*- coding: utf-8 -*-

# To run:
# python 01_ingest_spam_dedup.py --input comments_cleaned.parquet --out stage1_clean.parquet


import re, os, argparse, hashlib
import pandas as pd
import numpy as np

URL_RE = re.compile(r"(https?://|www\.)\S+", re.I)
DOMAIN_RE = re.compile(r"\b[\w-]+\.(com|net|org|io|co|shop|store|xyz|club|info)\b", re.I)
INVITE_RE = re.compile(r"\b(subscribe|follow|promo|discount|sale|giveaway|coupon|code|deal|dm|inbox|whatsapp|wa\.me|telegram|click|buy now|order)\b", re.I)
CONTACT_RE = re.compile(r"(\+\d{6,}|\b\d{3,}[-\s]?\d{3,}[-\s]?\d{3,}\b|@[A-Za-z0-9_.-]+)", re.I)
REPEAT_RE = re.compile(r"(.)\1{3,}")  # e.g., loooove!!!!!!
EMOJI_RE = re.compile(
    "["                     # pragmatic emoji cluster
    "\U0001F300-\U0001F6FF"
    "\U0001F900-\U0001FAFF"
    "\u2600-\u26FF\u2700-\u27BF"
    "]", flags=re.UNICODE
)

def normalize_text(s: str) -> str:
    if not isinstance(s, str): s = "" if s is None else str(s)
    try:
        import ftfy
        s = ftfy.fix_text(s)
    except Exception:
        pass
    s = s.replace("\u200b", " ")
    s = re.sub(r"\s+", " ", s.strip())
    return s

def simple_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def spam_rules(row: str) -> bool:
    txt = row
    lc = txt.lower()
    has_url = bool(URL_RE.search(lc) or DOMAIN_RE.search(lc))
    has_inv = bool(INVITE_RE.search(lc) or CONTACT_RE.search(lc))
    many_emoji = len(EMOJI_RE.findall(txt)) >= 4
    long_repeat = bool(REPEAT_RE.search(txt))
    too_short = len(lc.split()) < 2
    return (has_url and has_inv) or long_repeat or (many_emoji and has_inv) or (has_url and too_short)

def load_any(path: str, text_candidates=("clean_text","textOriginal","text","comment")):
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    # pick text column
    for c in text_candidates:
        if c in df.columns:
            text_col = c
            break
    else:
        obj_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c])]
        if not obj_cols:
            raise SystemExit("No text-like column found.")
        text_col = obj_cols[0]
    # time column (optional, used later)
    tcol = None
    for c in ["publishedAt","created_at","createdAt","time","date","timestamp"]:
        if c in df.columns:
            tcol = c
            break
    return df, text_col, tcol

def main():
    ap = argparse.ArgumentParser(description="Ingest → normalize → de-dup → spam filter")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="stage1_clean.parquet")
    args = ap.parse_args()

    df, text_col, tcol = load_any(args.input)
    df = df.copy()
    df["__text__"] = df[text_col].map(normalize_text)

    # exact de-dup (on normalized)
    df["__dupkey__"] = df["__text__"].map(lambda s: simple_hash(s))
    dup_mask = df["__dupkey__"].duplicated(keep="first")
    df["is_exact_dup"] = dup_mask

    # spam rules
    df["is_spam_rule"] = df["__text__"].map(spam_rules)

    # a few helpful meta features (for later analysis or ML spam, optional)
    df["len_words"] = df["__text__"].str.split().map(len)
    df["url_count"] = df["__text__"].str.count(URL_RE.pattern)
    df["emoji_count"] = df["__text__"].str.count(EMOJI_RE.pattern)

    # keep only non-spam, non-dup
    clean = df.loc[~df["is_spam_rule"] & ~df["is_exact_dup"]].copy()

    # tidy types
    for c in ["is_spam_rule","is_exact_dup"]:
        clean[c] = clean[c].astype(bool)
    if tcol:
        clean["publishedAt"] = pd.to_datetime(df[tcol], errors="coerce")

    # save + a small summary
    clean.to_parquet(args.out, index=False)
    total = len(df); kept = len(clean)
    print(f"[ingest] total={total:,} kept={kept:,} ({kept/total:.1%}) | "
          f"spam={df['is_spam_rule'].sum():,} dup={df['is_exact_dup'].sum():,}")
    # optional CSV summary
    pd.DataFrame({
        "total":[total],
        "kept":[kept],
        "spam":[int(df["is_spam_rule"].sum())],
        "dup":[int(df["is_exact_dup"].sum())]
    }).to_csv("out_ingest_summary.csv", index=False)

if __name__ == "__main__":
    main()
