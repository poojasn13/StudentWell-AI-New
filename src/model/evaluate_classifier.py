"""
Phase 4 — DistilBERT Classifier Evaluation
===========================================
Loads the fine-tuned model and evaluates it on the held-out test set.

Outputs
-------
logs/evaluation_report.txt          — full metrics report
assets/confusion_matrix.png         — confusion matrix heatmap
assets/per_class_metrics.png        — per-class precision / recall / F1 bar chart
logs/test_predictions.csv           — raw predictions for inspection

Run
---
    python src/model/evaluate_classifier.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR      = Path("models/distilbert_emotion")
PROCESSED_DIR  = Path("data/processed")
LABEL_MAP_FILE = PROCESSED_DIR / "label_map.json"
EVAL_REPORT    = Path("logs/evaluation_report.txt")
CM_PLOT        = Path("assets/confusion_matrix.png")
PER_CLASS_PLOT = Path("assets/per_class_metrics.png")
PRED_CSV       = Path("logs/test_predictions.csv")
LOG_FILE       = Path("logs/evaluation.log")

MAX_SEQ_LEN = 256
BATCH_SIZE  = 32

CANONICAL_EMOTIONS = ["neutral", "joy", "sadness", "anger", "fear", "surprise", "disgust"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

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
# Device
# ---------------------------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ---------------------------------------------------------------------------
# Dataset wrapper (same as training)
# ---------------------------------------------------------------------------

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length):
        self.dataset   = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item  = self.dataset[idx]
        text  = item["text"]
        label = int(item["label"])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
            "text":           text,
        }

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray, labels: list):
    Path("assets").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True", fontsize=13)
    ax.set_title("DistilBERT Emotion Classifier — Confusion Matrix (Test Set)", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(CM_PLOT), dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Confusion matrix saved to {CM_PLOT}")


def plot_per_class_metrics(report_dict: dict, labels: list):
    precisions = [report_dict[l]["precision"] for l in labels]
    recalls    = [report_dict[l]["recall"]    for l in labels]
    f1s        = [report_dict[l]["f1-score"]  for l in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precisions, width, label="Precision", color="#2196F3")
    ax.bar(x,         recalls,    width, label="Recall",    color="#4CAF50")
    ax.bar(x + width, f1s,        width, label="F1",        color="#FF5722")

    ax.set_xlabel("Emotion Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics — DistilBERT Emotion Classifier", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(PER_CLASS_PLOT), dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Per-class metrics chart saved to {PER_CLASS_PLOT}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate():
    if not MODEL_DIR.exists():
        log.error(f"Model not found at {MODEL_DIR}")
        log.error("Run src/model/train_classifier.py first.")
        return

    device = get_device()
    log.info(f"Using device: {device}")

    # Load label map
    with open(LABEL_MAP_FILE) as f:
        label_data = json.load(f)
    id2label = {int(k): v for k, v in label_data["id2label"].items()}
    label_names = [id2label[i] for i in range(len(id2label))]

    # Load model + tokenizer
    log.info(f"Loading model from {MODEL_DIR}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_DIR))
    model     = DistilBertForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model     = model.to(device)
    model.eval()

    # Load test set
    test_hf = load_from_disk(str(PROCESSED_DIR / "test"))
    test_ds = EmotionDataset(test_hf, tokenizer, MAX_SEQ_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    log.info(f"Test set size: {len(test_ds)} rows")

    # Run inference
    all_preds  = []
    all_labels = []
    all_probs  = []
    all_texts  = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"]

            outputs  = model(input_ids=input_ids, attention_mask=attention_mask)
            logits   = outputs.logits.cpu()
            probs    = torch.softmax(logits, dim=-1).numpy()
            preds    = logits.argmax(dim=-1).numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())
            all_texts.extend(batch["text"])

    # Compute metrics
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average="micro",    zero_division=0)
    w_f1     = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    report   = classification_report(
        all_labels, all_preds, target_names=label_names, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        all_labels, all_preds, target_names=label_names, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    # Confidence statistics
    max_probs = np.max(all_probs, axis=1)
    avg_conf  = float(np.mean(max_probs))
    low_conf_count = int((max_probs < 0.5).sum())

    # Build report text
    lines = []
    lines.append("=" * 60)
    lines.append("STUDENTWELL AI — EMOTION CLASSIFIER EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append(f"Test set size    : {len(all_labels)} rows")
    lines.append(f"Model            : distilbert-base-uncased (fine-tuned)")
    lines.append(f"Device           : {device}")
    lines.append("")
    lines.append("--- Overall Metrics ---")
    lines.append(f"  Accuracy       : {acc:.4f}  ({acc*100:.2f}%)")
    lines.append(f"  Macro F1       : {macro_f1:.4f}")
    lines.append(f"  Micro F1       : {micro_f1:.4f}")
    lines.append(f"  Weighted F1    : {w_f1:.4f}")
    lines.append(f"  Avg Confidence : {avg_conf:.4f}")
    lines.append(f"  Low-conf preds : {low_conf_count} ({low_conf_count/len(all_labels)*100:.1f}% < 0.50)")
    lines.append("")
    lines.append("--- Per-Class Report ---")
    lines.append(report_str)
    lines.append("--- Confusion Matrix ---")
    lines.append("  (Rows = True labels, Columns = Predicted)")
    header = "         " + "  ".join(f"{n[:6]:>6}" for n in label_names)
    lines.append(header)
    for i, row in enumerate(cm):
        row_str = f"  {label_names[i]:<8}" + "  ".join(f"{v:>6}" for v in row)
        lines.append(row_str)
    lines.append("")
    lines.append("--- Confidence Distribution ---")
    for threshold in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pct = (max_probs >= threshold).mean() * 100
        lines.append(f"  ≥ {threshold:.1f} confidence : {pct:.1f}% of predictions")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    print("\n" + report_text)

    # Save report
    EVAL_REPORT.parent.mkdir(exist_ok=True)
    with open(EVAL_REPORT, "w") as f:
        f.write(report_text)
    log.info(f"Evaluation report saved to {EVAL_REPORT}")

    # Save predictions CSV
    pred_label_names = [id2label[p] for p in all_preds]
    true_label_names = [id2label[l] for l in all_labels]
    df_preds = pd.DataFrame({
        "text":           all_texts,
        "true_emotion":   true_label_names,
        "pred_emotion":   pred_label_names,
        "confidence":     max_probs,
        "correct":        [p == l for p, l in zip(pred_label_names, true_label_names)],
    })
    # Add per-class probability columns
    for i, em in enumerate(label_names):
        df_preds[f"prob_{em}"] = [p[i] for p in all_probs]
    df_preds.to_csv(PRED_CSV, index=False)
    log.info(f"Predictions saved to {PRED_CSV}")

    # Plots
    plot_confusion_matrix(cm, label_names)
    plot_per_class_metrics(report, label_names)

    log.info("Evaluation complete.")
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "avg_confidence": avg_conf,
    }


if __name__ == "__main__":
    evaluate()
