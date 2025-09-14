# NaNSense — Beauty Comment Intelligence (L’Oréal × Monash Datathon)

Pipeline for transforming large-scale social/UGC comments into **pillars** (skincare/makeup/haircare/fragrance), **sub‑categories** (e.g., Foundation, Serum), **emotions** (joy/neutral/sadness/anger), **consumer pain points** (Price, Packaging, Quality, Availability, Sustainability, Compatibility), plus **QER** (quality engagement ratio) and **trend signals**.

> **Purpose**: help brand/marketing teams see where conversations are growing, which sub‑categories & pain points matter, and how emotion + engagement evolve over time.

---

## ✨ Key Features

- **Robust ingest**: text normalisation, de‑duplication, rule‑based spam filter.
- **Weakly‑supervised spam ML**: HashingVectorizer + SGD, precision‑tuned threshold.
- **Pillar + subcategory rules**: high‑precision regex over curated beauty dictionary.
- **Emotion inference**: DistilRoBERTa model (`j-hartmann/emotion-english-distilroberta-base`) mapped 6→4 labels with a neutral gate.
- **Pain‑point tagging (regex)**: conservative patterns to avoid currency‑only false positives.
- **QER**: combines length normaliser, emotion weight, and log‑likes.
- **Trends**: weekly/monthly “z‑burst” term risers per pillar + trendlines.
- **CSV artifacts** for dashboards (categories, pains, trendlines, QER summary).

---

## 🗂️ Repo Structure

```
.
├─ 01_ingest_spam_dedup.py         # normalise → dedup → basic spam rules
├─ 01b_spam_ml.py                  # weakly-supervised ML spam classifier
├─ 02_pillar_and_emotion_plus.py   # pillars + subcategories + emotions
├─ 03_qer_and_trends_plus.py       # QER, trends, pain-points, exports
├─ data/                           # (you create) raw & intermediate inputs
├─ out_qer/                        # QER summaries
├─ out_trends/                     # top risers + trendlines
├─ out_categories/                 # category breakdowns
├─ out_pain/                       # pain point tables
└─ README.md
```

---

## 🧰 Requirements

- Python **3.10+** (tested on 3.10/3.11/3.12)
- Recommended: **CUDA‑enabled GPU** for emotion inference (PyTorch + Transformers)
- Install dependencies:

```bash
pip install -U pandas numpy scikit-learn scipy torch transformers evaluate ftfy pyarrow
```

> **Windows note**: Hugging Face cache symlink warnings are safe. Enable **Developer Mode** for faster caching if desired.

---

## 🚀 Quickstart

Assume your cleaned raw comments (CSV/Parquet) live at `data/comments_cleaned.parquet`
with a text column like `clean_text` (scripts auto‑detect from common names).

1) **Ingest + de‑dup + rule spam**
```bash
python 01_ingest_spam_dedup.py --input data/comments_cleaned.parquet --out stage1_clean.parquet
```

2) **Spam ML (weakly‑supervised)**
```bash
python 01b_spam_ml.py --raw data/comments_cleaned.parquet --clean stage1_clean.parquet \
  --out-prefix stage1b --sample 120000 --target-precision 0.95
# Outputs:
# - stage1b_scored.parquet (raw + scores)
# - stage1b_clean_ml.parquet (final cleaned)
# - out_spam_ml/metrics.txt (AP, chosen threshold, report)
```

3) **Pillars + subcategories + emotions**
```bash
python 02_pillar_and_emotion_plus.py --input stage1b_clean_ml.parquet \
  --out stage2_tagged_plus.parquet --batch-size 192 --max-length 192
```
> GPU auto‑detected. On CPU this step is slower.

4) **QER, Trends, Pain points, Category tables**
```bash
python 03_qer_and_trends_plus.py --input stage2_tagged_plus.parquet --period W --min-df 50
# Exports:
# out_qer/qer_summary.csv
# out_categories/category_breakdown_by_pillar.csv
# out_pain/pain_counts_by_pillar.csv
# out_pain/pain_counts_by_pillar_period.csv
# out_trends/top_risers_by_pillar.csv
# out_trends/term_trendlines.csv
```

---

## 📦 Inputs & Outputs

**Inputs**
- Parquet/CSV with a text column (auto‑detected from: `clean_text`, `textOriginal`, `text`, `comment`).
- Optional: `publishedAt`/`created_at` timestamp for trends; `likeCount` for QER.

**Core Output (stage 2)**  
`stage2_tagged_plus.parquet` includes:
- `is_skincare`, `is_makeup`, `is_haircare`, `is_fragrance`
- `pillar_primary`, `category_primary`, `category_all`
- `emotion_base` (joy/neutral/sadness/anger), `emotion_conf`, `p_joy`…`p_anger`
- (plus your original meta like `publishedAt`, `likeCount` if present)

**CSV Artifacts**
- `out_qer/qer_summary.csv` — overall and per‑pillar QER mean & counts
- `out_categories/category_breakdown_by_pillar.csv`
- `out_pain/pain_counts_by_pillar.csv` + `_by_pillar_period.csv`
- `out_trends/top_risers_by_pillar.csv`, `term_trendlines.csv`

---

## 🧮 Method Details

- **Spam ML**: HashingVectorizer (char 3–5, word 1–2) → TF‑IDF → SGD (log loss), `class_weight="balanced"`. Threshold tuned for target precision on silver validation.
- **Pillars/Subcats**: curated dictionaries + regex unions; hair‑vs‑skin disambiguation; primary pillar via hit count.
- **Emotions**: `j-hartmann/emotion-english-distilroberta-base` (6 logits) → 4‑label mapping with **neutral gate** to reduce neutral drift.
- **Pain points**: **revised** conservative regex — currency requires numbers (e.g., `RM59`, `$12`) and/or explicit sentiment cues (“too expensive”, “affordable”). Currency‑only mentions are **ignored**.
- **QER**: `(length_weight + emotion_weight) * log1p(likes)`; emotion weights: joy=+0.2, neutral=0, sadness=−0.1, anger=−0.2.
- **Trends**: weekly/monthly term counts on a fixed vocab; z‑score of term share vs rolling baseline (“z‑burst”).

---

## 🧪 Evaluation (how to)

- **Spam ML**: inspect `out_spam_ml/metrics.txt` (Average Precision, classification report).
- **Emotions**: (Optional) use your 240‑row labelled set to compute accuracy/F1; adjust neutral gate if needed.
- **Pillars/Subcats**: spot‑check precision on random samples; tune dictionaries.
- **Pain points**: verify coverage < 100% and “Price” no longer dominates due to currency‑only matches.

> Keep a small `notebooks/` folder for sampling and error analysis (FP/FN review).

---

## ⚙️ Configuration & Flags

- All scripts support `--help`.
- Common flags:
  - `--period {W,M}` for trends aggregation.
  - `--min-df` minimum document frequency for trend vocab.
  - `--likes-col` (default `likeCount`) for QER.
  - `--batch-size` and `--max-length` for emotion inference.

---

## 🩹 Troubleshooting

- **Transformers dtype warning**: handled in code; GPU/CPU dtype auto‑set.
- **Windows symlink warning**: safe; enable Developer Mode for faster caching.
- **CUDA not used**: install CUDA‑enabled PyTorch and verify `torch.cuda.is_available()` is `True`.
- **Everything tagged as Price / pain coverage 100%**: ensure you’re on the **revised pain regex** (currency‑only no longer counts).

---

## 🔒 Data & Privacy

- Don’t commit raw data. Use `.gitignore`:
```
# data & heavy intermediates
data/
*.parquet
out_spam_ml/
__pycache__/
*.ipynb_checkpoints/
```
- Only commit aggregated CSVs that contain no PII.

---

## 🗺️ Roadmap

- Subcategory‑aware pain mapping (e.g., “cakey” → Makeup/Foundation).
- Lightweight supervised pain classifier using the 240‑row set.
- Sarcasm robustness & domain‑adapted emotion fine‑tune.
- Streamlit/Power BI dashboard hooked to `out_*` CSVs.

---

## 🙏 Acknowledgements

- L’Oréal × Monash Datathon organisers & mentors.
- Pretrained model: `j-hartmann/emotion-english-distilroberta-base` (Hugging Face).

---

## 📫 Contact

Questions, bugs, or ideas? Open an issue or reach out via GitHub.
