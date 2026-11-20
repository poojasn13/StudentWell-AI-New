"""
Phase 12 — Analytics Module
=============================
Provides data and chart functions for the analytics dashboard.
Reads from:
  - SQLite (student ratings) via rating_store
  - Streamlit session_state conversation logs (live session data)

All charts use Matplotlib (as required by the project spec).
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.evaluation.rating_store import get_all_ratings, get_summary_stats

log = logging.getLogger(__name__)

CANONICAL_EMOTIONS = ["neutral", "joy", "sadness", "anger", "fear", "surprise", "disgust"]

EMOTION_COLORS = {
    "neutral":  "#90A4AE",
    "joy":      "#FFD54F",
    "sadness":  "#64B5F6",
    "anger":    "#EF5350",
    "fear":     "#CE93D8",
    "surprise": "#80DEEA",
    "disgust":  "#A5D6A7",
}


# ---------------------------------------------------------------------------
# Chart helpers — all return a Matplotlib Figure
# ---------------------------------------------------------------------------

def chart_emotion_distribution(emotion_counts: dict[str, int]) -> plt.Figure:
    """Bar chart of emotion distribution from current session or dataset."""
    emotions = [e for e in CANONICAL_EMOTIONS if e in emotion_counts]
    counts   = [emotion_counts.get(e, 0) for e in emotions]
    colors   = [EMOTION_COLORS.get(e, "#BDBDBD") for e in emotions]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(emotions, counts, color=colors, edgecolor="white", linewidth=0.8)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(count), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Emotion Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(counts) * 1.2 + 1 if counts else 5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def chart_confidence_distribution(confidences: list[float]) -> plt.Figure:
    """Histogram of prediction confidence scores."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(confidences, bins=20, range=(0, 1), color="#42A5F5", edgecolor="white", alpha=0.85)
    ax.axvline(x=0.45, color="#EF5350", linestyle="--", linewidth=1.5, label="Low-conf threshold (0.45)")
    ax.set_title("Emotion Prediction Confidence Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def chart_latency_comparison(latencies_normal: list[float], latencies_aware: list[float]) -> plt.Figure:
    """Box plot comparing response latency between both systems."""
    fig, ax = plt.subplots(figsize=(7, 4))
    data = []
    labels = []
    if latencies_normal:
        data.append(latencies_normal)
        labels.append("Normal\nChatbot")
    if latencies_aware:
        data.append(latencies_aware)
        labels.append("Emotion-Aware\nChatbot")

    if not data:
        ax.text(0.5, 0.5, "No latency data yet", ha="center", va="center", transform=ax.transAxes)
    else:
        bp = ax.boxplot(data, patch_artist=True, labels=labels)
        colors = ["#90CAF9", "#A5D6A7"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

    ax.set_title("Response Latency Comparison (ms)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Latency (ms)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def chart_ratings_comparison(stats: dict) -> plt.Figure:
    """
    Grouped bar chart comparing the four rating dimensions between
    the normal and emotion-aware chatbot.
    """
    dimensions = ["empathy", "relevance", "usefulness", "tone"]
    normal_vals = [stats["normal"].get(d, 0)        for d in dimensions]
    aware_vals  = [stats["emotion_aware"].get(d, 0)  for d in dimensions]

    x     = np.arange(len(dimensions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, normal_vals, width, label="Normal Chatbot",       color="#90CAF9", edgecolor="white")
    bars2 = ax.bar(x + width/2, aware_vals,  width, label="Emotion-Aware Chatbot", color="#A5D6A7", edgecolor="white")

    # Value labels
    for bar in bars1 + bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.05, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in dimensions])
    ax.set_ylim(0, 5.5)
    ax.axhline(y=5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_title("Student Ratings: Normal vs Emotion-Aware Chatbot", fontsize=13, fontweight="bold")
    ax.set_ylabel("Average Rating (1–5)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def chart_emotion_over_time(turns: list[dict]) -> plt.Figure:
    """
    Line chart showing how emotion changes across conversation turns.
    turns: list of {"turn_number": int, "emotion": str}
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    if not turns:
        ax.text(0.5, 0.5, "No conversation data yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Emotion Across Conversation Turns", fontsize=13, fontweight="bold")
        plt.tight_layout()
        return fig

    emotion_to_y = {e: i for i, e in enumerate(CANONICAL_EMOTIONS)}
    turn_nums   = [t["turn_number"] for t in turns]
    emotions    = [t["emotion"] for t in turns]
    y_vals      = [emotion_to_y.get(e, 0) for e in emotions]
    colors      = [EMOTION_COLORS.get(e, "#BDBDBD") for e in emotions]

    ax.plot(turn_nums, y_vals, color="#78909C", linewidth=1, zorder=1)
    ax.scatter(turn_nums, y_vals, c=colors, s=80, zorder=2, edgecolors="white", linewidths=0.8)

    ax.set_yticks(range(len(CANONICAL_EMOTIONS)))
    ax.set_yticklabels(CANONICAL_EMOTIONS)
    ax.set_xlabel("Turn Number")
    ax.set_title("Emotion Across Conversation Turns", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)

    # Legend patches
    patches = [mpatches.Patch(color=EMOTION_COLORS[e], label=e) for e in CANONICAL_EMOTIONS]
    ax.legend(handles=patches, loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    return fig


def chart_safety_triggers(crisis_count: int, concern_count: int, safe_count: int) -> plt.Figure:
    """Pie chart of safety trigger breakdown."""
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Safe", "Concern", "Crisis"]
    values = [safe_count, concern_count, crisis_count]
    colors_pie = ["#A5D6A7", "#FFD54F", "#EF5350"]

    # Filter out zero slices
    non_zero = [(l, v, c) for l, v, c in zip(labels, values, colors_pie) if v > 0]
    if not non_zero:
        ax.text(0.5, 0.5, "No safety data yet", ha="center", va="center", transform=ax.transAxes)
    else:
        ls, vs, cs = zip(*non_zero)
        ax.pie(vs, labels=ls, colors=cs, autopct="%1.0f%%", startangle=90,
               wedgeprops={"edgecolor": "white", "linewidth": 1.5})

    ax.set_title("Safety Check Outcomes", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Data aggregation helpers (called by the Streamlit dashboard page)
# ---------------------------------------------------------------------------

def build_session_analytics(session_data: dict) -> dict:
    """
    Build analytics dict from a serialised ConversationManager.to_dict() payload.
    Returns a dict ready for the dashboard.
    """
    turns = session_data.get("turns", [])
    if not turns:
        return {"total_turns": 0, "emotions": {}, "confidences": [], "latencies": [], "safety": {}}

    emotions    = {}
    confidences = []
    latencies   = []
    safety_counts = {"safe": 0, "concern": 0, "crisis": 0}

    for t in turns:
        em = t.get("emotion", "neutral")
        emotions[em] = emotions.get(em, 0) + 1
        conf = t.get("confidence", 0.0)
        if conf > 0:
            confidences.append(conf)
        lat = t.get("latency_ms", 0.0)
        if lat > 0:
            latencies.append(lat)
        sl = t.get("safety_level", "safe")
        safety_counts[sl] = safety_counts.get(sl, 0) + 1

    return {
        "total_turns":    len(turns),
        "emotions":       emotions,
        "confidences":    confidences,
        "latencies":      latencies,
        "safety_counts":  safety_counts,
        "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
    }


if __name__ == "__main__":
    # Quick test — generate and save sample charts
    Path("assets").mkdir(exist_ok=True)

    fig = chart_emotion_distribution({"neutral": 30, "joy": 25, "fear": 20, "anger": 15, "sadness": 10, "disgust": 5, "surprise": 8})
    fig.savefig("assets/sample_emotion_dist.png", dpi=100)
    plt.close(fig)
    print("Sample charts generated in assets/")
