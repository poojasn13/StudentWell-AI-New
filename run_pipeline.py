"""
Full Pipeline Runner
====================
Runs all preprocessing and training steps in order after dataset generation
is complete.

Steps
-----
1. validate_dataset   — clean raw CSV
2. preprocess         — tokenize and split by conversation_id
3. train_classifier   — fine-tune DistilBERT on MPS
4. evaluate_classifier — test-set evaluation and charts

Usage
-----
    python run_pipeline.py

Or run individual steps:
    python run_pipeline.py --step validate
    python run_pipeline.py --step preprocess
    python run_pipeline.py --step train
    python run_pipeline.py --step evaluate
"""

import argparse
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run_validate():
    log.info("=" * 50)
    log.info("STEP 1: Dataset Validation")
    log.info("=" * 50)
    from src.data.validate_dataset import validate_dataset
    df = validate_dataset()
    if df is None:
        log.error("Validation failed — stopping pipeline")
        sys.exit(1)
    log.info(f"Validation complete: {len(df)} clean rows")


def run_preprocess():
    log.info("=" * 50)
    log.info("STEP 2: Preprocessing & Splitting")
    log.info("=" * 50)
    from src.data.preprocess import preprocess
    result = preprocess()
    if result is None:
        log.error("Preprocessing failed — stopping pipeline")
        sys.exit(1)
    log.info("Preprocessing complete")


def run_train():
    log.info("=" * 50)
    log.info("STEP 3: DistilBERT Training")
    log.info("=" * 50)
    from src.model.train_classifier import train
    train()


def run_evaluate():
    log.info("=" * 50)
    log.info("STEP 4: Evaluation on Test Set")
    log.info("=" * 50)
    from src.model.evaluate_classifier import evaluate
    metrics = evaluate()
    if metrics:
        log.info(f"Test Accuracy : {metrics['accuracy']:.4f}")
        log.info(f"Test Macro F1 : {metrics['macro_f1']:.4f}")


STEPS = {
    "validate":  run_validate,
    "preprocess": run_preprocess,
    "train":     run_train,
    "evaluate":  run_evaluate,
}


def main():
    parser = argparse.ArgumentParser(description="StudentWell AI — training pipeline")
    parser.add_argument(
        "--step",
        choices=list(STEPS.keys()),
        default=None,
        help="Run a single step (default: run all steps in order)",
    )
    args = parser.parse_args()

    if args.step:
        STEPS[args.step]()
    else:
        log.info("Running full pipeline: validate → preprocess → train → evaluate")
        for step_name, step_fn in STEPS.items():
            step_fn()
        log.info("Full pipeline complete.")


if __name__ == "__main__":
    main()
