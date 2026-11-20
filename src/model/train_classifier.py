"""
Phase 4 — DistilBERT Conversational Emotion Classifier — Training
==================================================================
Fine-tunes DistilBERT (distilbert-base-uncased) for 7-class emotion
classification on the student conversation dataset.

Design decisions
----------------
- Uses Apple Silicon MPS device automatically when available.
- Input format: "[CONVERSATION CONTEXT] <ctx> [CURRENT MESSAGE] <text>"
  This matches what preprocess.py builds in the model_input / text field.
- Max sequence length 256 — long enough for context+message, short enough
  for M4 memory.  Most inputs are well under this limit.
- Batch size 16 on MPS (safe for M4 8–16 GB unified memory).
- 5 training epochs with early stopping (patience 2 on val loss).
- Learning rate 2e-5 with linear warmup over first 10% of steps.
- Class-weighted loss to handle minor label imbalance.
- Saves the best checkpoint by validation F1.
- Saves training/val loss curves as a PNG.

Outputs
-------
models/distilbert_emotion/          — final model + tokenizer
models/distilbert_emotion_best/     — best checkpoint by val F1
logs/training.log                   — full training log
assets/training_loss_curve.png      — loss visualisation

Run
---
    python src/model/train_classifier.py

After training completes, run:
    python src/model/evaluate_classifier.py
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_from_disk
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROCESSED_DIR   = Path("data/processed")
MODEL_SAVE_DIR  = Path("models/distilbert_emotion")
BEST_CKPT_DIR   = Path("models/distilbert_emotion_best")
LABEL_MAP_FILE  = PROCESSED_DIR / "label_map.json"
LOSS_CURVE_FILE = Path("assets/training_loss_curve.png")
LOG_FILE        = Path("logs/training.log")

PRETRAINED_MODEL = "distilbert-base-uncased"

# Training hyperparameters — tuned for MacBook Air M4
MAX_SEQ_LEN  = 256
BATCH_SIZE   = 16
NUM_EPOCHS   = 5
LEARNING_RATE = 2e-5
WARMUP_RATIO  = 0.1          # fraction of total steps used for warmup
WEIGHT_DECAY  = 0.01
EARLY_STOP_PATIENCE = 2      # stop if val F1 does not improve for N epochs
SEED         = 42

NUM_LABELS = 7

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
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        log.info("Using Apple Silicon MPS device")
        return torch.device("mps")
    if torch.cuda.is_available():
        log.info("Using CUDA device")
        return torch.device("cuda")
    log.info("Using CPU (no GPU available)")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Torch Dataset wrapper
# ---------------------------------------------------------------------------

class EmotionDataset(TorchDataset):
    """Wraps a HuggingFace dataset and tokenizes on-the-fly."""

    def __init__(self, hf_dataset, tokenizer, max_length: int):
        self.dataset    = hf_dataset
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item  = self.dataset[idx]
        text  = item["text"]   # already the formatted model_input string
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
        }


# ---------------------------------------------------------------------------
# Class weights for imbalanced labels
# ---------------------------------------------------------------------------

def compute_class_weights(hf_dataset, num_labels: int) -> torch.Tensor:
    counts = np.zeros(num_labels)
    for item in hf_dataset:
        counts[int(item["label"])] += 1
    # Inverse-frequency weighting
    total = counts.sum()
    weights = total / (num_labels * counts)
    weights = weights / weights.sum() * num_labels  # normalise
    log.info(f"Class weights: {weights.round(3).tolist()}")
    return torch.tensor(weights, dtype=torch.float)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits
            loss    = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = get_device()

    # Load label map
    with open(LABEL_MAP_FILE) as f:
        label_data = json.load(f)
    label2id = label_data["label2id"]
    id2label = {int(k): v for k, v in label_data["id2label"].items()}
    log.info(f"Label mapping: {label2id}")

    # Load tokenizer
    log.info(f"Loading tokenizer: {PRETRAINED_MODEL}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(PRETRAINED_MODEL)

    # Add special tokens for our context/message format
    special_tokens = ["[CONVERSATION CONTEXT]", "[CURRENT MESSAGE]"]
    num_added = tokenizer.add_tokens(special_tokens, special_tokens=True)
    log.info(f"Added {num_added} special tokens to tokenizer")

    # Load HuggingFace datasets
    log.info("Loading processed datasets...")
    train_hf = load_from_disk(str(PROCESSED_DIR / "train"))
    val_hf   = load_from_disk(str(PROCESSED_DIR / "val"))
    log.info(f"Train: {len(train_hf)} rows | Val: {len(val_hf)} rows")

    # Torch datasets
    train_ds = EmotionDataset(train_hf, tokenizer, MAX_SEQ_LEN)
    val_ds   = EmotionDataset(val_hf,   tokenizer, MAX_SEQ_LEN)

    # DataLoaders
    # num_workers=0 is safest on MPS (avoids multiprocessing issues)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    log.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Model
    log.info(f"Loading model: {PRETRAINED_MODEL}")
    model = DistilBertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    )
    # Resize embeddings to include our special tokens
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {total_params:,} total | {trainable_params:,} trainable")

    # Class-weighted loss
    class_weights = compute_class_weights(train_hf, NUM_LABELS).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Scheduler — linear warmup then linear decay
    total_steps  = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    log.info(f"Total training steps: {total_steps} | Warmup steps: {warmup_steps}")

    # Training history
    history = {
        "train_loss": [], "val_loss": [],
        "val_acc": [], "val_macro_f1": []
    }

    best_val_f1    = -1.0
    patience_count = 0

    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    BEST_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("STARTING TRAINING")
    log.info("=" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        correct    = 0
        total      = 0

        for step, batch in enumerate(train_loader, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = criterion(outputs.logits, labels)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            preds  = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            if step % 50 == 0 or step == len(train_loader):
                running_acc = correct / total
                log.info(
                    f"Epoch {epoch}/{NUM_EPOCHS} | Step {step}/{len(train_loader)} "
                    f"| Loss: {epoch_loss/step:.4f} | Acc: {running_acc:.4f}"
                )

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, criterion)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        log.info(
            f"\n{'='*50}\n"
            f"Epoch {epoch} Summary:\n"
            f"  Train Loss : {avg_train_loss:.4f}\n"
            f"  Val   Loss : {val_loss:.4f}\n"
            f"  Val   Acc  : {val_acc:.4f}\n"
            f"  Val Macro F1: {val_f1:.4f}\n"
            f"{'='*50}"
        )

        # Save best checkpoint
        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            patience_count = 0
            model.save_pretrained(str(BEST_CKPT_DIR))
            tokenizer.save_pretrained(str(BEST_CKPT_DIR))
            log.info(f"  ★ New best checkpoint saved (val F1: {best_val_f1:.4f})")
        else:
            patience_count += 1
            log.info(f"  No improvement. Patience: {patience_count}/{EARLY_STOP_PATIENCE}")
            if patience_count >= EARLY_STOP_PATIENCE:
                log.info(f"Early stopping triggered at epoch {epoch}")
                break

    # Save final model (best weights are in BEST_CKPT_DIR)
    # Load best and re-save as the main model
    log.info("Loading best checkpoint for final save...")
    best_model = DistilBertForSequenceClassification.from_pretrained(
        str(BEST_CKPT_DIR),
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    )
    best_model.resize_token_embeddings(len(tokenizer))
    best_model.save_pretrained(str(MODEL_SAVE_DIR))
    tokenizer.save_pretrained(str(MODEL_SAVE_DIR))
    log.info(f"Final model saved to {MODEL_SAVE_DIR}")

    # Save training history JSON
    history_path = Path("logs/training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Plot loss curves
    plot_loss_curves(history)

    log.info("=" * 60)
    log.info("TRAINING COMPLETE")
    log.info(f"Best validation Macro F1 : {best_val_f1:.4f}")
    log.info(f"Model saved to           : {MODEL_SAVE_DIR}")
    log.info(f"Loss curve saved to      : {LOSS_CURVE_FILE}")
    log.info("=" * 60)

    print("\n--- Training Summary ---")
    print(f"Best Val Macro F1 : {best_val_f1:.4f}")
    print(f"Model saved to    : {MODEL_SAVE_DIR}")
    print(f"Next step         : python src/model/evaluate_classifier.py")


# ---------------------------------------------------------------------------
# Loss curve plot
# ---------------------------------------------------------------------------

def plot_loss_curves(history: dict):
    LOSS_CURVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train Loss",  color="#2196F3")
    axes[0].plot(epochs, history["val_loss"],   marker="s", label="Val Loss",    color="#FF5722")
    axes[0].set_title("Training & Validation Loss", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Macro F1
    axes[1].plot(epochs, history["val_macro_f1"], marker="^", label="Val Macro F1", color="#4CAF50")
    axes[1].plot(epochs, history["val_acc"],       marker="d", label="Val Accuracy", color="#9C27B0")
    axes[1].set_title("Validation Macro F1 & Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("DistilBERT Emotion Classifier — Training Curves", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(LOSS_CURVE_FILE), dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Loss curve saved to {LOSS_CURVE_FILE}")


if __name__ == "__main__":
    train()
