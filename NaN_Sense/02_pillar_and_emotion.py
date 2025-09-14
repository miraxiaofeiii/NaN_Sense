#!/usr/bin/env python
# -*- coding: utf-8 -*-

# To run:
# python 02_pillar_and_emotion.py --input stage1b_clean_ml.parquet --out stage2_tagged.parquet --batch-size 192 --max-length 192

import os, argparse, re, json
import pandas as pd
import torch

# ---------------- Pillar dictionary ----------------
BEAUTY_DICT = {
    "skincare": ["skincare","lip balm","serum","moisturizer","lotion","balm","oil","cleanser",
                 "toner","mask","peel","hydrating","dryness","oily","sensitive","acne","pimple",
                 "blemish","dark spot","hyperpigmentation","wrinkle","fine line","spf","sunscreen",
                 "retinol","vitamin c","hyaluronic","collagen","niacinamide","salicylic","glycolic",
                 "ceramide","skin barrier","barrier repair"],
    "makeup": ["makeup","make up","foundation","concealer","powder","primer","contour","blush",
               "bronzer","highlight","mascara","eyeliner","eyeshadow","palette","brow","lashes",
               "lipstick","lip gloss","lip liner","matte","dewy","setting spray","setting powder",
               "bake","cakey","crease","bb cream","cc cream","filter","airbrush"],
    "haircare": ["hair","shampoo","conditioner","serum","oil","hair mask","hair oil","leave-in","treatment",
                 "frizz","hair spray","hair gel","mousse","scalp","dandruff",
                 "hair fall","split ends","keratin","perm","bleach","dye","balayage","ombre"],
    "fragrance": ["perfume","fragrance","cologne","eau de parfum","eau de toilette","scent",
                  "woody","floral","citrus","musky","spicy","amber","vanilla","sillage","projection","edt","edp"],
    "loreal_brands": ["loreal","l'oréal","l’oreal","loreal paris","revitalift","maybelline",
                      "garnier","kiehls","lancome","nyx","ysl","giorgio armani","ralph lauren"]
}

# ---------------- Subcategory dictionary ----------------
SUBCAT_DICT = {
    "makeup": {
        "Lipstick": ["lipstick"],
        "Foundation": ["foundation"],
        "Mascara": ["mascara"],
        "Eyeshadow": ["eyeshadow","palette"],
        "Eyeliner": ["eyeliner"],
        "Blush": ["blush"],
        "Primer": ["primer"],
        "Concealer": ["concealer"],
        "Powder": ["powder","setting powder"],
        "Brow": ["brow","brow pencil","brow gel"],
        "Setting Spray": ["setting spray"],
        "Contour": ["contour"],
        "Highlighter": ["highlight","highlighter"],
        "Lip Gloss": ["lip gloss","gloss"],
        "Lip Liner": ["lip liner"],
        "BB/CC Cream": ["bb cream","cc cream"],
    },
    "skincare": {
        "Cleanser": ["cleanser","face wash"],
        "Toner": ["toner"],
        "Serum": ["serum","ampoule","essence"],
        "Moisturizer": ["moisturizer","lotion","cream","barrier repair","skin barrier"],
        "Sunscreen": ["sunscreen","spf","sunblock"],
        "Mask": ["mask","sheet mask"],
        "Exfoliator": ["exfoliator","peel","scrub","glycolic","salicylic"],
        "Eye Cream": ["eye cream"],
        "Acne Treatment": ["acne","pimple","blemish","dark spot","hyperpigmentation"],
        "Actives": ["retinol","niacinamide","hyaluronic","collagen","ceramide","vitamin c"],
    },
    "haircare": {
        "Shampoo": ["shampoo"],
        "Conditioner": ["conditioner"],
        "Hair Oil": ["hair oil","oil"],
        "Hair Mask": ["hair mask","mask"],
        "Leave-in": ["leave-in"],
        "Treatment": ["treatment","keratin"],
        "Hairspray": ["hair spray"],
        "Hair Gel": ["hair gel","gel","mousse"],
        "Scalp": ["scalp","dandruff","hair fall","split ends"],
        "Dye/Color": ["dye","color","bleach","balayage","ombre","perm"],
        "Serum": ["hair serum","serum"],
    },
    "fragrance": {
        "Perfume": ["perfume","fragrance","scent"],
        "Eau de Parfum": ["eau de parfum","edp"],
        "Eau de Toilette": ["eau de toilette","edt"],
        "Cologne": ["cologne"],
        "Body Mist": ["body mist","mist","rollerball"],
    }
}

def regex_union(terms):
    return rf"(?<!\w)(?:{'|'.join([re.escape(t).replace(r'\ ', r'\s+') for t in terms])})(?!\w)"

def build_patterns():
    return {k: re.compile(regex_union(v), re.I) for k,v in BEAUTY_DICT.items()}

def add_pillars(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["__lc__"] = df["__text__"].str.lower().str.replace(r"\s+"," ", regex=True)
    pats = build_patterns()
    for cat, pat in pats.items():
        df[f"is_{cat}"] = df["__lc__"].str.contains(pat, na=False)
    # override skincare when hair context
    hair_ctx = df["__lc__"].str.contains(r"\bhair\b", na=False)
    bad_overlap = df["__lc__"].str.contains(r"\b(?:smooth|shine|shiny|glow)\b", na=False)
    df.loc[hair_ctx & bad_overlap, "is_haircare"] = True
    df.loc[hair_ctx & bad_overlap, "is_skincare"] = False
    df["is_ambiguous_skin"] = False
    return df

# ---------------- Subcategory tagging ----------------
def _compile_subcat_regex():
    compiled = {}
    for pillar, mapping in SUBCAT_DICT.items():
        compiled[pillar] = {name: re.compile(regex_union(phrases), re.I) for name, phrases in mapping.items()}
    return compiled

_SUBCAT_RX = _compile_subcat_regex()
PILLAR_ORDER = ["makeup","skincare","haircare","fragrance"]

def _detect_primary_pillar(row) -> str | None:
    hits = [p for p in PILLAR_ORDER if row.get(f"is_{p}", False)]
    if len(hits) == 1:
        return hits[0]
    text = str(row.get("__text__", "")).lower()
    best, best_n = None, 0
    for p in PILLAR_ORDER:
        rx_map = _SUBCAT_RX[p]
        n = sum(bool(rx.search(text)) for rx in rx_map.values())
        if n > best_n:
            best, best_n = p, n
    return best

def add_subcategories(df: pd.DataFrame) -> pd.DataFrame:
    dff = df.copy()
    prim, cats_all, pillars = [], [], []
    for _, row in dff.iterrows():
        pillar = _detect_primary_pillar(row) or "makeup"
        text = str(row["__text__"])
        rx_map = _SUBCAT_RX[pillar]
        hits = [name for name, rx in rx_map.items() if rx.search(text)]
        prim.append(hits[0] if hits else "Other")
        cats_all.append(hits if hits else ["Other"])
        pillars.append(pillar or "unknown")
    dff["category_primary"] = prim
    dff["category_all"] = cats_all
    dff["pillar_primary"] = pillars
    return dff

# ---------------- Emotions (fine-tuned 4-class loader + fallback) ----------------
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_emotion_model():
    ckpt = "emotion_model_beauty_4class"
    if os.path.exists(ckpt):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(
            ckpt,
            dtype=(torch.float16 if device()=="cuda" else torch.float32)
        ).to(device())
        T = 1.0
        temp_path = os.path.join(ckpt, "temperature.json")
        if os.path.exists(temp_path):
            try:
                T = float(json.load(open(temp_path))["temperature"])
            except Exception:
                pass
        return tok, mdl, T, ["joy","neutral","sadness","anger"]
    else:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base", use_fast=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(
            "j-hartmann/emotion-english-distilroberta-base",
            dtype=(torch.float16 if device()=="cuda" else torch.float32)
        ).to(device())
        return tok, mdl, 1.0, None

def add_emotions(df: pd.DataFrame, max_length=192, batch_size=192) -> pd.DataFrame:
    """
    Predict emotions for __text__ and add:
      - emotion_base (str in {joy, neutral, sadness, anger})
      - emotion_conf (float)
      - p_joy, p_neutral, p_sadness, p_anger (floats)
    Works with either a fine-tuned 4-class checkpoint or the 6-class fallback.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch as _torch

    dev = device()
    tok, mdl, TEMP, labelset = load_emotion_model()  # labelset==None -> 6-class fallback

    try:
        mdl = mdl.to_bettertransformer()
    except Exception:
        pass
    mdl.eval()

    condensed, conf = [], []
    p = {"joy": [], "neutral": [], "sadness": [], "anger": []}

    @_torch.no_grad()
    def infer(batch_txt):
        enc = tok(
            batch_txt, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt"
        ).to(dev)
        logits = mdl(**enc).logits / TEMP
        probs = logits.softmax(-1).cpu().tolist()

        for row in probs:
            # --- build "four" probs no matter which backend we’re on ---
            if labelset:  # fine-tuned 4-class: [joy, neutral, sadness, anger]
                joy, neu, sad, ang = row
                four = {"joy": joy, "neutral": neu, "sadness": sad, "anger": ang}
                # choose label (argmax)
                pred = max(four, key=four.get)
            else:         # fallback 6-class → collapse to 4
                id2label = mdl.config.id2label
                six = {id2label[i].lower(): row[i] for i in range(len(row))}
                anger = six.get("anger", 0.0) + six.get("disgust", 0.0) + six.get("fear", 0.0)
                sadness = six.get("sadness", 0.0)
                joy = six.get("joy", 0.0)
                neutral = six.get("neutral", 0.0)
                four = {"joy": joy, "neutral": neutral, "sadness": sadness, "anger": anger}
                # neutral gate only for fallback
                m_non_neu = max(joy, sadness, anger)
                pred = "neutral" if m_non_neu < 0.45 else max(four, key=four.get)

            # --- append exactly once ---
            condensed.append(pred)
            conf.append(four[pred])
            p["joy"].append(four["joy"])
            p["neutral"].append(four["neutral"])
            p["sadness"].append(four["sadness"])
            p["anger"].append(four["anger"])

    texts = df["__text__"].astype(str).tolist()
    for i in range(0, len(texts), batch_size):
        infer(texts[i:i + batch_size])

    # --- safety check ---
    n = len(df)
    if not (len(condensed) == len(conf) == len(p["joy"]) == len(p["neutral"]) == len(p["sadness"]) == len(p["anger"]) == n):
        raise RuntimeError(
            f"[add_emotions] length mismatch: df={n} | "
            f"base={len(condensed)} conf={len(conf)} "
            f"p_joy={len(p['joy'])} p_neutral={len(p['neutral'])} "
            f"p_sadness={len(p['sadness'])} p_anger={len(p['anger'])}"
        )

    out = df.copy()
    out["emotion_base"] = condensed
    out["emotion_conf"] = conf
    out["p_joy"] = p["joy"]
    out["p_neutral"] = p["neutral"]
    out["p_sadness"] = p["sadness"]
    out["p_anger"] = p["anger"]
    return out


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Pillar tagging + Subcategories + Emotion inference")
    ap.add_argument("--input", default="stage1_clean.parquet")
    ap.add_argument("--out", default="stage2_tagged.parquet")
    ap.add_argument("--batch-size", type=int, default=192)
    ap.add_argument("--max-length", type=int, default=192)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    if "__text__" not in df.columns:
        raise SystemExit("Input missing __text__ column")

    df = add_pillars(df)
    df = add_subcategories(df)
    df = add_emotions(df, max_length=args.max_length, batch_size=args.batch_size)

    # tighten dtypes
    for c in [c for c in df.columns if c.startswith("is_")]:
        df[c] = df[c].astype(bool)
    for c in ["p_joy","p_neutral","p_sadness","p_anger","emotion_conf"]:
        if c in df: df[c] = df[c].astype("float32")

    df.to_parquet(args.out, index=False)
    print(f"[tag+subcat+emotion] saved {args.out} rows={len(df):,}")

if __name__ == "__main__":
    main()
