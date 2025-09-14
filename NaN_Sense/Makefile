.PHONY: all ingest spam_ml tag qer_trends clean help

# --------------------------
# Config (override on CLI)
# --------------------------
DATA            ?= data/comments_cleaned.parquet
STAGE1          ?= stage1_clean.parquet
OUTPREFIX       ?= stage1b
STAGE1B_CLEAN   ?= $(OUTPREFIX)_clean_ml.parquet
STAGE2          ?= stage2_tagged_plus.parquet

# QER/Trends
PERIOD          ?= W          # W=weekly, M=monthly
MIN_DF          ?= 50         # min doc freq for trend vocab
LIKES_COL       ?= likeCount

# Emotion inference
BATCH           ?= 192
MAXLEN          ?= 192

# Spam-ML
SAMPLE          ?= 120000
TARGET_PREC     ?= 0.95

# --------------------------
# Default: run everything
# --------------------------
all: ingest spam_ml tag qer_trends

help:
\t@echo ""
\t@echo "NaNSense — Beauty Comment Intelligence Pipeline"
\t@echo "Usage:"
\t@echo "  make all                 # run full pipeline"
\t@echo "  make ingest              # stage1: normalize/dedup/rule-spam"
\t@echo "  make spam_ml             # stage1b: weakly supervised spam ML"
\t@echo "  make tag                 # stage2: pillars + subcats + emotions"
\t@echo "  make qer_trends          # QER + trends + pain/category exports"
\t@echo "  make clean               # remove intermediates/outputs"
\t@echo ""
\t@echo "Override defaults, e.g.:"
\t@echo "  make all DATA=data/yt.parquet PERIOD=M MIN_DF=30"
\t@echo ""

# --------------------------
# Stage 1: ingest
# --------------------------
ingest: $(STAGE1)

$(STAGE1): 01_ingest_spam_dedup.py $(DATA)
\tpython 01_ingest_spam_dedup.py --input $(DATA) --out $(STAGE1)

# --------------------------
# Stage 1b: spam ML
# --------------------------
spam_ml: $(STAGE1B_CLEAN)

$(STAGE1B_CLEAN): 01b_spam_ml.py $(DATA) $(STAGE1)
\tpython 01b_spam_ml.py --raw $(DATA) --clean $(STAGE1) --out-prefix $(OUTPREFIX) --sample $(SAMPLE) --target-precision $(TARGET_PREC)

# --------------------------
# Stage 2: pillars + emotions
# --------------------------
tag: $(STAGE2)

$(STAGE2): 02_pillar_and_emotion_plus.py $(STAGE1B_CLEAN)
\tpython 02_pillar_and_emotion_plus.py --input $(STAGE1B_CLEAN) --out $(STAGE2) --batch-size $(BATCH) --max-length $(MAXLEN)

# --------------------------
# Stage 3: QER + trends + exports
# --------------------------
qer_trends: out_qer/qer_summary.csv out_categories/category_breakdown_by_pillar.csv out_pain/pain_counts_by_pillar.csv out_trends/top_risers_by_pillar.csv

# Running the trends/export script creates all CSVs below in one go.
out_qer/qer_summary.csv out_categories/category_breakdown_by_pillar.csv out_pain/pain_counts_by_pillar.csv out_trends/top_risers_by_pillar.csv: 03_qer_and_trends_plus.py $(STAGE2)
\tpython 03_qer_and_trends_plus.py --input $(STAGE2) --period $(PERIOD) --min-df $(MIN_DF) --likes-col $(LIKES_COL)

# --------------------------
# Clean
# --------------------------
clean:
\trm -rf $(STAGE1) $(OUTPREFIX)_scored.parquet $(OUTPREFIX)_clean_ml.parquet $(STAGE2) \\\
\t  out_qer/ out_trends/ out_categories/ out_pain/ out_spam_ml/
\t@echo "Cleaned intermediates and outputs."
