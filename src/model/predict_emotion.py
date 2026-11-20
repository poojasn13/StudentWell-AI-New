"""
Phase 4 — Emotion Prediction Module
=====================================
Reusable module for predicting emotion from conversational text.

Usage
-----
    from src.model.predict_emotion import EmotionPredictor

    predictor = EmotionPredictor()   # loads model once

    result = predictor.predict(
        text="I have an exam tomorrow and I still haven't started studying.",
        context="Student mentioned feeling behind on their coursework."
    )
    print(result)
    # {
    #   "emotion": "fear",
    #   "confidence": 0.87,
    #   "probabilities": {"neutral": 0.03, "joy": 0.01, ...},
    #   "low_confidence": False,
    #   "label_id": 4
    # }

Design
------
- Model is loaded once and reused across calls (singleton pattern).
- Input format matches training: [CONVERSATION CONTEXT] ... [CURRENT MESSAGE] ...
- Low-confidence threshold is 0.45 — predictions below this are flagged.
- Thread-safe for single-threaded Streamlit use.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR      = Path("models/distilbert_emotion")
LABEL_MAP_FILE = Path("data/processed/label_map.json")

CONTEXT_TOKEN       = "[CONVERSATION CONTEXT]"
MESSAGE_TOKEN       = "[CURRENT MESSAGE]"
EMPTY_CONTEXT       = "No prior context."
MAX_SEQ_LEN         = 256
LOW_CONF_THRESHOLD  = 0.45

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class EmotionPredictor:
    """
    Loads the fine-tuned DistilBERT model and provides emotion prediction.
    Instantiate once per application session for efficiency.
    """

    _instance = None  # module-level singleton

    def __init__(self, model_dir: Optional[Path] = None, device: Optional[str] = None):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.device    = self._select_device(device)
        self.model     = None
        self.tokenizer = None
        self.label2id  = {}
        self.id2label  = {}
        self._load()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @staticmethod
    def _select_device(device_str: Optional[str]) -> torch.device:
        if device_str:
            return torch.device(device_str)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load(self):
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_dir}. "
                "Run src/model/train_classifier.py first."
            )

        log.info(f"Loading emotion model from {self.model_dir} on {self.device}")

        # Label map — try model dir first, fall back to processed dir
        label_map_path = self.model_dir / "label_map.json"
        if not label_map_path.exists():
            label_map_path = LABEL_MAP_FILE
        with open(label_map_path) as f:
            data = json.load(f)
        self.label2id = data["label2id"]
        self.id2label = {int(k): v for k, v in data["id2label"].items()}

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(self.model_dir))
        self.model     = DistilBertForSequenceClassification.from_pretrained(
            str(self.model_dir)
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Also save label map next to model for portability
        label_map_dest = self.model_dir / "label_map.json"
        if not label_map_dest.exists():
            import shutil
            shutil.copy(str(LABEL_MAP_FILE), str(label_map_dest))

        log.info("Emotion model loaded successfully.")

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def _build_input(self, text: str, context: str) -> str:
        ctx = context.strip() if context.strip() else EMPTY_CONTEXT
        return f"{CONTEXT_TOKEN} {ctx} {MESSAGE_TOKEN} {text.strip()}"

    def predict(self, text: str, context: str = "") -> dict:
        """
        Predict emotion for a student message.

        Parameters
        ----------
        text : str
            The current student message.
        context : str, optional
            Summary or previous turns of the conversation.

        Returns
        -------
        dict with keys:
            emotion        : str    — predicted emotion label
            confidence     : float  — probability of predicted class
            probabilities  : dict   — {emotion: probability} for all 7 classes
            low_confidence : bool   — True if confidence < LOW_CONF_THRESHOLD
            label_id       : int    — integer label index
        """
        if not text or not text.strip():
            return self._neutral_fallback("Empty input text")

        model_input = self._build_input(text, context)

        encoding = self.tokenizer(
            model_input,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits[0]
            probs   = F.softmax(logits, dim=-1).cpu().numpy()

        label_id   = int(probs.argmax())
        emotion    = self.id2label[label_id]
        confidence = float(probs[label_id])

        probabilities = {
            self.id2label[i]: round(float(p), 4)
            for i, p in enumerate(probs)
        }

        return {
            "emotion":        emotion,
            "confidence":     round(confidence, 4),
            "probabilities":  probabilities,
            "low_confidence": confidence < LOW_CONF_THRESHOLD,
            "label_id":       label_id,
            "model_input":    model_input,  # for debugging only
        }

    def predict_batch(self, items: list[dict]) -> list[dict]:
        """
        Predict emotions for a list of {"text": ..., "context": ...} dicts.
        More efficient than calling predict() in a loop for large batches.
        """
        if not items:
            return []

        inputs = [self._build_input(it.get("text", ""), it.get("context", "")) for it in items]

        encodings = self.tokenizer(
            inputs,
            max_length=MAX_SEQ_LEN,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = F.softmax(outputs.logits, dim=-1).cpu().numpy()

        results = []
        for i, row_probs in enumerate(probs):
            label_id   = int(row_probs.argmax())
            emotion    = self.id2label[label_id]
            confidence = float(row_probs[label_id])
            results.append({
                "emotion":        emotion,
                "confidence":     round(confidence, 4),
                "probabilities":  {self.id2label[j]: round(float(p), 4) for j, p in enumerate(row_probs)},
                "low_confidence": confidence < LOW_CONF_THRESHOLD,
                "label_id":       label_id,
            })
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _neutral_fallback(reason: str) -> dict:
        log.warning(f"Falling back to neutral prediction: {reason}")
        return {
            "emotion":        "neutral",
            "confidence":     0.0,
            "probabilities":  {e: 0.0 for e in
                               ["neutral","joy","sadness","anger","fear","surprise","disgust"]},
            "low_confidence": True,
            "label_id":       0,
        }

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_predictor_instance: Optional[EmotionPredictor] = None


def get_predictor() -> EmotionPredictor:
    """Return a module-level singleton predictor (loads once per process)."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = EmotionPredictor()
    return _predictor_instance


def predict_emotion(text: str, context: str = "") -> dict:
    """
    Module-level convenience wrapper.
    Uses the singleton predictor — safe for use from Streamlit.
    """
    return get_predictor().predict(text, context)


# ---------------------------------------------------------------------------
# Quick test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Loading emotion predictor...")
    p = EmotionPredictor()

    test_cases = [
        {
            "context": "",
            "text": "I have an exam tomorrow and I haven't studied at all.",
            "expected": "fear",
        },
        {
            "context": "Student was worried about failing.",
            "text": "I finally passed the exam! I can't believe it!",
            "expected": "joy",
        },
        {
            "context": "Student was struggling with a group project.",
            "text": "Nobody in my group is doing any work. It's so frustrating.",
            "expected": "anger",
        },
        {
            "context": "",
            "text": "What resources does the library have for research?",
            "expected": "neutral",
        },
        {
            "context": "Student put a lot of effort into an assignment.",
            "text": "I got a D on that paper after working on it all night. I'm so disappointed.",
            "expected": "sadness",
        },
    ]

    print("\n--- Prediction Test ---\n")
    for tc in test_cases:
        result = p.predict(tc["text"], tc["context"])
        match  = "✓" if result["emotion"] == tc["expected"] else "✗"
        print(f"{match} Expected: {tc['expected']:<10} | Got: {result['emotion']:<10} "
              f"| Conf: {result['confidence']:.3f} | Low: {result['low_confidence']}")
        print(f"  Text: {tc['text'][:70]}")
        print(f"  Probs: {result['probabilities']}")
        print()
