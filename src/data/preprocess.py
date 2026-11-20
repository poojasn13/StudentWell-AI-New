"""
Phase 3 — Dataset Preprocessing
=================================
Tokenizes the cleaned dataset and splits it into train / val / test sets.

Key design decisions
--------------------
1.  Split is done by conversation_id — entire conversations stay together
    so there is NO data leakage between train and test.
2.  Input format combines context + text into a single string using
    special delimiter tokens that DistilBERT will learn from:
        [CONVERSATION CONTEXT] <context> [CURRENT MESSAGE] <text>
    If context is empty the CONTEXT section uses "No prior context."
3.  Labels are integer-encoded using a fixed mapping stored alongside
    the processed data so the same mapping is used at inference time.
4.  The final split ratio is 70 / 15 / 15 (train / val / test).
5.  Saves HuggingFace Dataset objects as Arrow files for fast loading
    during training, plus CSV versions for inspection.

Outputs
-------
data/processed/
  label_map.json          — {label: int} mapping (also saved as int2label)
  train.csv               — training rows with model_input column
  val.csv                 — validation rows
  test.csv                — test rows
  train/                  — HuggingFace Arrow dataset
  val/                    — HuggingFace Arrow dataset
  test/                   — HuggingFace Arrow dataset
  split_report.txt        — distribution per split

Run
---
    python src/data/preprocess.py
"""

import json
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from datasets import Dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLEAN_FILE      = Path("data/raw/student_conversations_clean.csv")
PROCESSED_DIR   = Path("data/processed")
LABEL_MAP_FILE  = PROCESSED_DIR / "label_map.json"
SPLIT_REPORT    = PROCESSED_DIR / "split_report.txt"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Fixed label order — must match training and inference
CANONICAL_EMOTIONS = ["neutral", "joy", "sadness", "anger", "fear", "surprise", "disgust"]
LABEL2ID = {label: idx for idx, label in enumerate(CANONICAL_EMOTIONS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

CONTEXT_TOKEN  = "[CONVERSATION CONTEXT]"
MESSAGE_TOKEN  = "[CURRENT MESSAGE]"
EMPTY_CONTEXT  = "No prior context."

LOG_FILE = Path("logs/preprocess.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Input formatting
# ---------------------------------------------------------------------------

def build_model_input(context, text: str) -> str:
    """
    Combine conversation context and current student message into
    a single string for DistilBERT tokenisation.

    Format:
        [CONVERSATION CONTEXT] <context text> [CURRENT MESSAGE] <student message>

    Handles None / NaN context values (turn 1 rows have no context).
    """
    import pandas as pd
    # Guard against NaN / None / float values from CSV loading
    if context is None or (isinstance(context, float) and pd.isna(context)):
        context = ""
    ctx = str(context).strip() if str(context).strip() else EMPTY_CONTEXT
    return f"{CONTEXT_TOKEN} {ctx} {MESSAGE_TOKEN} {str(text).strip()}"


# ---------------------------------------------------------------------------
# Split by conversation_id
# ---------------------------------------------------------------------------

def split_by_conversation(df: pd.DataFrame, seed: int = 42):
    """
    Split conversation IDs into train / val / test, then assign rows.
    This prevents any conversation from spanning multiple splits (leakage prevention).
    """
    rng = np.random.default_rng(seed)

    conv_ids = df["conversation_id"].unique()
    rng.shuffle(conv_ids)

    n = len(conv_ids)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    # test gets the remainder to avoid off-by-one
    n_test  = n - n_train - n_val

    train_ids = set(conv_ids[:n_train])
    val_ids   = set(conv_ids[n_train:n_train + n_val])
    test_ids  = set(conv_ids[n_train + n_val:])

    df_train = df[df["conversation_id"].isin(train_ids)].copy()
    df_val   = df[df["conversation_id"].isin(val_ids)].copy()
    df_test  = df[df["conversation_id"].isin(test_ids)].copy()

    log.info(f"Split: {len(train_ids)} train convs / {len(val_ids)} val convs / {len(test_ids)} test convs")
    log.info(f"Rows : {len(df_train)} train / {len(df_val)} val / {len(df_test)} test")

    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def preprocess():
    if not CLEAN_FILE.exists():
        log.error(f"Clean dataset not found: {CLEAN_FILE}")
        log.error("Run src/data/validate_dataset.py first.")
        return

    log.info(f"Loading clean dataset from {CLEAN_FILE}")
    df = pd.read_csv(CLEAN_FILE)
    log.info(f"Loaded {len(df)} rows, {df['conversation_id'].nunique()} conversations")

    # Build model_input column
    df["model_input"] = df.apply(
        lambda row: build_model_input(row["context"], row["text"]), axis=1
    )
    df["label"] = df["emotion"].map(LABEL2ID)

    # Sanity check for unmapped labels
    unmapped = df["label"].isna().sum()
    if unmapped > 0:
        log.warning(f"{unmapped} rows have unmapped labels — dropping them")
        df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)

    # Split
    df_train, df_val, df_test = split_by_conversation(df)

    # Save processed dir
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Save label maps
    with open(LABEL_MAP_FILE, "w") as f:
        json.dump({"label2id": LABEL2ID, "id2label": {str(k): v for k, v in ID2LABEL.items()}}, f, indent=2)
    log.info(f"Label map saved to {LABEL_MAP_FILE}")

    # Save CSVs for human inspection
    for split_name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        path = PROCESSED_DIR / f"{split_name}.csv"
        split_df.to_csv(path, index=False)
        log.info(f"Saved {split_name}.csv ({len(split_df)} rows)")

    # Save HuggingFace Arrow datasets (faster loading during training)
    hf_cols = ["model_input", "label", "conversation_id", "turn_id", "emotion"]
    for split_name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        hf_df = split_df[hf_cols].rename(columns={"model_input": "text"})
        hf_dataset = Dataset.from_pandas(hf_df, preserve_index=False)
        save_path = PROCESSED_DIR / split_name
        hf_dataset.save_to_disk(str(save_path))
        log.info(f"Saved HuggingFace dataset → {save_path}")

    # Build split report
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("STUDENTWELL AI — PREPROCESSING SPLIT REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Total rows         : {len(df)}")
    report_lines.append(f"Total conversations: {df['conversation_id'].nunique()}")
    report_lines.append(f"Label mapping      : {LABEL2ID}")
    report_lines.append("")

    for split_name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        n_convs = split_df["conversation_id"].nunique()
        n_rows  = len(split_df)
        pct     = n_rows / len(df) * 100
        report_lines.append(f"--- {split_name.upper()} ({pct:.1f}%) ---")
        report_lines.append(f"  Conversations : {n_convs}")
        report_lines.append(f"  Rows          : {n_rows}")
        dist = Counter(split_df["emotion"])
        for em in CANONICAL_EMOTIONS:
            cnt = dist.get(em, 0)
            report_lines.append(f"    {em:<12} {cnt:>6}")
        report_lines.append("")

    report_lines.append("--- MODEL INPUT SAMPLE (first 3 training rows) ---")
    for _, row in df_train.head(3).iterrows():
        report_lines.append(f"  Label : {row['emotion']} ({row['label']})")
        report_lines.append(f"  Input : {row['model_input'][:120]}...")
        report_lines.append("")

    report_lines.append("=" * 60)

    report_text = "\n".join(report_lines)
    with open(SPLIT_REPORT, "w") as f:
        f.write(report_text)
    log.info(f"Split report saved to {SPLIT_REPORT}")

    print("\n" + report_text)
    log.info("Preprocessing complete.")
    return df_train, df_val, df_test


if __name__ == "__main__":
    preprocess()
