"""
Phase 2 — Dataset Validation and Cleaning
==========================================
Reads the raw generated CSV, applies quality checks, removes bad rows,
and writes a clean CSV ready for preprocessing.

Checks performed
----------------
1.  Missing values in required fields (text, emotion, conversation_id, turn_id)
2.  Empty or whitespace-only text
3.  Text that is too short (< 8 characters)
4.  Invalid emotion labels (not in canonical set)
5.  Exact duplicate texts (case-insensitive)
6.  Near-duplicate texts (Jaccard similarity > 0.85 on word n-grams)
7.  Conversation integrity (turn IDs must start at 1 and be sequential)
8.  Label distribution report (flags if any emotion is below 80% of target)
9.  Emotion keyword leakage check (warns if the emotion word appears in the text)

Outputs
-------
data/raw/student_conversations_clean.csv    — cleaned dataset
logs/validation_report.txt                 — full quality report

Run
---
    python src/data/validate_dataset.py
"""

import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_FILE   = Path("data/raw/student_conversations.csv")
CLEAN_FILE = Path("data/raw/student_conversations_clean.csv")
REPORT_FILE = Path("logs/validation_report.txt")

CANONICAL_EMOTIONS = {
    "neutral", "joy", "sadness", "anger", "fear", "surprise", "disgust"
}

EMOTION_TARGETS = {
    "neutral":  572,
    "joy":      572,
    "sadness":  572,
    "anger":    572,
    "fear":     572,
    "surprise": 572,
    "disgust":  572,
}

MIN_TEXT_LENGTH   = 8    # characters
NEAR_DUP_THRESHOLD = 0.85  # Jaccard similarity on 3-grams

LOG_FILE = Path("logs/validation.log")
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
# Helpers
# ---------------------------------------------------------------------------

def word_ngrams(text: str, n: int = 3) -> set:
    """Return set of word n-grams for a text string."""
    words = re.sub(r"[^a-z0-9\s]", "", text.lower()).split()
    if len(words) < n:
        return set(words)
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


# ---------------------------------------------------------------------------
# Check functions — each returns a boolean Series (True = keep)
# ---------------------------------------------------------------------------

def check_missing(df: pd.DataFrame, report: list) -> pd.Series:
    required = ["conversation_id", "turn_id", "text", "emotion"]
    mask = pd.Series(True, index=df.index)
    for col in required:
        bad = df[col].isna() | (df[col].astype(str).str.strip() == "")
        count = bad.sum()
        report.append(f"  Missing/empty '{col}'           : {count} rows removed")
        mask &= ~bad
    return mask


def check_text_length(df: pd.DataFrame, report: list) -> pd.Series:
    mask = df["text"].str.strip().str.len() >= MIN_TEXT_LENGTH
    removed = (~mask).sum()
    report.append(f"  Text too short (< {MIN_TEXT_LENGTH} chars)       : {removed} rows removed")
    return mask


def check_invalid_labels(df: pd.DataFrame, report: list) -> pd.Series:
    mask = df["emotion"].isin(CANONICAL_EMOTIONS)
    removed = (~mask).sum()
    bad_labels = df.loc[~mask, "emotion"].unique().tolist()
    report.append(f"  Invalid emotion labels             : {removed} rows removed — {bad_labels}")
    return mask


def check_exact_duplicates(df: pd.DataFrame, report: list) -> pd.Series:
    key = df["text"].str.lower().str.strip()
    mask = ~key.duplicated(keep="first")
    removed = (~mask).sum()
    report.append(f"  Exact duplicate texts              : {removed} rows removed")
    return mask


def check_near_duplicates(df: pd.DataFrame, report: list) -> pd.Series:
    """
    Remove near-duplicates using Jaccard similarity on 3-grams.
    O(n²) in the worst case — only practical for < 50k rows.
    Uses a bucketing heuristic: compare only within same emotion bucket.
    """
    keep = pd.Series(True, index=df.index)
    removed_total = 0

    for emotion in CANONICAL_EMOTIONS:
        sub = df[keep & (df["emotion"] == emotion)].copy()
        if len(sub) < 2:
            continue

        texts     = sub["text"].str.lower().str.strip().tolist()
        indices   = sub.index.tolist()
        ngram_sets = [word_ngrams(t) for t in texts]
        to_remove  = set()

        for i in range(len(texts)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(texts)):
                if j in to_remove:
                    continue
                sim = jaccard(ngram_sets[i], ngram_sets[j])
                if sim >= NEAR_DUP_THRESHOLD:
                    to_remove.add(j)

        for local_idx in to_remove:
            keep[indices[local_idx]] = False
            removed_total += 1

    report.append(f"  Near-duplicate texts (Jaccard≥{NEAR_DUP_THRESHOLD:.2f}): {removed_total} rows removed")
    return keep


def check_conversation_integrity(df: pd.DataFrame, report: list) -> tuple[pd.Series, dict]:
    """
    For each conversation, verify turn_ids are 1-based integers.
    Conversations with gaps or starting at values other than 1 are flagged.
    Returns a keep mask (we don't remove whole conversations, just warn) and stats.
    """
    keep = pd.Series(True, index=df.index)
    bad_convs = []

    for conv_id, group in df.groupby("conversation_id"):
        turns = sorted(group["turn_id"].tolist())
        expected = list(range(1, len(turns) + 1))
        if turns != expected:
            bad_convs.append(conv_id)

    stats = {"bad_conversations": len(bad_convs), "examples": bad_convs[:5]}
    report.append(f"  Conversations with broken turn seq : {len(bad_convs)} (not removed, flagged only)")
    return keep, stats


def check_emotion_keyword_leakage(df: pd.DataFrame, report: list):
    """Warn (not remove) when the emotion label word appears verbatim in the text."""
    leakage_counts = {}
    for emotion in CANONICAL_EMOTIONS:
        sub  = df[df["emotion"] == emotion]
        leaks = sub["text"].str.lower().str.contains(r"\b" + emotion + r"\b", regex=True).sum()
        leakage_counts[emotion] = int(leaks)

    report.append("  Emotion keyword leakage (warnings — rows NOT removed):")
    for em, cnt in leakage_counts.items():
        pct = cnt / max(1, (df["emotion"] == em).sum()) * 100
        report.append(f"    {em:<10} {cnt:>5} rows ({pct:.1f}%)")
    return leakage_counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate_dataset():
    if not RAW_FILE.exists():
        log.error(f"Raw dataset not found: {RAW_FILE}")
        log.error("Run src/data/generate_dataset.py first.")
        return

    log.info(f"Loading raw dataset from {RAW_FILE}")
    df = pd.read_csv(RAW_FILE)
    initial_count = len(df)
    log.info(f"Initial row count: {initial_count}")

    report = []
    report.append("=" * 60)
    report.append("STUDENTWELL AI — DATASET VALIDATION REPORT")
    report.append("=" * 60)
    report.append(f"Raw dataset rows : {initial_count}")
    report.append("")
    report.append("--- Quality Checks ---")

    # Ensure text column is string
    df["text"]    = df["text"].astype(str)
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
    df["context"] = df["context"].fillna("").astype(str)

    # Apply checks in order
    mask = pd.Series(True, index=df.index)

    m1 = check_missing(df, report)
    mask &= m1

    m2 = check_text_length(df[mask], report)
    # Re-index m2 to full df
    full_m2 = pd.Series(True, index=df.index)
    full_m2[mask] = m2.values
    full_m2[~mask] = False
    mask &= full_m2

    m3 = check_invalid_labels(df[mask], report)
    full_m3 = pd.Series(True, index=df.index)
    full_m3[mask] = m3.values
    full_m3[~mask] = False
    mask &= full_m3

    m4 = check_exact_duplicates(df[mask], report)
    full_m4 = pd.Series(True, index=df.index)
    full_m4[mask] = m4.values
    full_m4[~mask] = False
    mask &= full_m4

    # Near-duplicate check (may take a minute on large datasets)
    log.info("Running near-duplicate detection (this may take a moment)...")
    m5 = check_near_duplicates(df[mask], report)
    full_m5 = pd.Series(True, index=df.index)
    full_m5[mask] = m5.values
    full_m5[~mask] = False
    mask &= full_m5

    # Integrity check (non-destructive)
    _, integrity_stats = check_conversation_integrity(df[mask], report)

    # Keyword leakage (non-destructive)
    check_emotion_keyword_leakage(df[mask], report)

    # Apply mask
    df_clean = df[mask].copy()
    df_clean.reset_index(drop=True, inplace=True)
    final_count = len(df_clean)

    report.append("")
    report.append("--- Label Distribution ---")
    report.append(f"{'Emotion':<12} {'Count':>7}  {'Target':>7}  {'Status'}")
    report.append("-" * 50)

    dist = df_clean["emotion"].value_counts().to_dict()
    total_ok = True
    for emotion in sorted(CANONICAL_EMOTIONS):
        actual = dist.get(emotion, 0)
        target = EMOTION_TARGETS[emotion]
        pct    = actual / target * 100 if target else 0
        status = "✓" if actual >= target else (f"⚠ {target-actual} short" if pct >= 80 else f"✗ CRITICALLY SHORT")
        if actual < target:
            total_ok = False
        report.append(f"  {emotion:<12} {actual:>7}  {target:>7}  {status}")

    report.append("")
    report.append("--- Summary ---")
    report.append(f"  Rows before cleaning : {initial_count}")
    report.append(f"  Rows removed         : {initial_count - final_count}")
    report.append(f"  Rows after cleaning  : {final_count}")
    total_target = sum(EMOTION_TARGETS.values())
    report.append(f"  Target total         : {total_target}")
    coverage = final_count / total_target * 100
    report.append(f"  Coverage             : {coverage:.1f}%")
    report.append(f"  Broken turn sequences: {integrity_stats['bad_conversations']}")
    if not total_ok:
        report.append("  ⚠ Some emotions are below target — consider running more generation.")
    else:
        report.append("  ✓ All emotion targets met.")
    report.append("=" * 60)

    # Save
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(CLEAN_FILE, index=False)
    log.info(f"Clean dataset saved to {CLEAN_FILE} ({final_count} rows)")

    # Write report
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(report)
    with open(REPORT_FILE, "w") as f:
        f.write(report_text)
    log.info(f"Validation report saved to {REPORT_FILE}")

    # Print to console
    print("\n" + report_text)
    return df_clean


if __name__ == "__main__":
    validate_dataset()
