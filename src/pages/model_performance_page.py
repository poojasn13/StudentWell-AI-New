"""
Streamlit Page — Model Performance (Phase 4 results display)
"""

import json
import streamlit as st
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def render():
    st.title("🎯 Model Performance")
    st.caption("DistilBERT emotion classifier evaluation results on the held-out test set.")

    eval_report = Path("logs/evaluation_report.txt")
    training_history = Path("logs/training_history.json")
    cm_plot     = Path("assets/confusion_matrix.png")
    per_class   = Path("assets/per_class_metrics.png")
    loss_curve  = Path("assets/training_loss_curve.png")

    if not eval_report.exists():
        st.warning(
            "Model has not been trained yet or evaluation has not been run.\n\n"
            "After dataset generation completes, run:\n"
            "```\npython run_pipeline.py\n```"
        )

        st.markdown("### Expected Results After Training")
        st.markdown("""
        Once training completes you will see:
        - **Accuracy** — overall classification accuracy on the test set
        - **Macro F1** — balanced F1 across all 7 emotion classes
        - **Per-class precision, recall, F1** — for each emotion
        - **Confusion matrix** — showing which emotions are confused
        - **Training/validation loss curves** — across epochs
        - **Confidence distribution** — prediction certainty histogram

        Results will reflect actual training on the generated dataset.
        No numbers are pre-filled or fabricated.
        """)
        return

    # Training history
    if training_history.exists():
        with open(training_history) as f:
            hist = json.load(f)
        best_f1  = max(hist.get("val_macro_f1", [0]))
        best_acc = max(hist.get("val_acc", [0]))
        epochs   = len(hist.get("train_loss", []))

        col1, col2, col3 = st.columns(3)
        col1.metric("Best Val Macro F1",  f"{best_f1:.4f}")
        col2.metric("Best Val Accuracy",  f"{best_acc:.4f}")
        col3.metric("Epochs Trained",     epochs)

    # Evaluation report text
    st.subheader("Test Set Evaluation Report")
    with open(eval_report) as f:
        report_text = f.read()
    st.code(report_text, language=None)

    # Plots
    if loss_curve.exists():
        st.subheader("Training & Validation Loss Curves")
        st.image(str(loss_curve), use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        if cm_plot.exists():
            st.subheader("Confusion Matrix")
            st.image(str(cm_plot), use_container_width=True)
    with col_right:
        if per_class.exists():
            st.subheader("Per-Class Precision / Recall / F1")
            st.image(str(per_class), use_container_width=True)

    # Dataset split report
    split_report = Path("data/processed/split_report.txt")
    if split_report.exists():
        st.subheader("Dataset Split Report")
        with open(split_report) as f:
            st.code(f.read(), language=None)
