"""
Streamlit Page — Analytics Dashboard (Phase 12)
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.evaluation.analytics import (
    chart_emotion_distribution,
    chart_confidence_distribution,
    chart_latency_comparison,
    chart_ratings_comparison,
    chart_emotion_over_time,
    chart_safety_triggers,
    build_session_analytics,
)
from src.evaluation.rating_store import get_summary_stats


def render():
    st.title("📊 Analytics Dashboard")
    st.caption("Live session analytics and student evaluation results. No fabricated data — all numbers reflect actual usage.")

    # --- Pull data from active sessions ---
    chat_manager = st.session_state.get("chat_manager")
    cmp_manager  = st.session_state.get("cmp_manager")

    session_data_chat = chat_manager.to_dict() if chat_manager else {"turns": []}
    session_data_cmp_b = cmp_manager.system_b.to_dict() if cmp_manager else {"turns": []}

    all_turns = session_data_chat["turns"] + session_data_cmp_b["turns"]
    analytics  = build_session_analytics({"turns": all_turns})

    # --- Top metrics row ---
    col1, col2, col3, col4, col5 = st.columns(5)
    total_turns    = analytics["total_turns"]
    safety_total   = sum(analytics["safety_counts"].values()) if total_turns else 0
    safety_trigger = (analytics["safety_counts"].get("crisis", 0) +
                      analytics["safety_counts"].get("concern", 0))

    col1.metric("Total Messages",      total_turns)
    col2.metric("Unique Emotions",     len(analytics["emotions"]))
    col3.metric("Avg Confidence",      f"{analytics['avg_confidence']:.0%}" if analytics['avg_confidence'] else "N/A")
    col4.metric("Avg Latency",         f"{analytics['avg_latency_ms']:.0f}ms" if analytics['avg_latency_ms'] else "N/A")
    col5.metric("Safety Triggers",     safety_trigger)

    st.divider()

    # --- Emotion distribution ---
    st.subheader("Emotion Distribution (Current Session)")
    if analytics["emotions"]:
        fig = chart_emotion_distribution(analytics["emotions"])
        st.pyplot(fig)
    else:
        st.info("Start chatting to see emotion distribution data.")

    # --- Two column section ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Confidence Distribution")
        if analytics["confidences"]:
            fig = chart_confidence_distribution(analytics["confidences"])
            st.pyplot(fig)
        else:
            st.caption("No confidence data yet.")

    with col_right:
        st.subheader("Safety Outcomes")
        sc = analytics["safety_counts"]
        fig = chart_safety_triggers(sc.get("crisis",0), sc.get("concern",0), sc.get("safe",0))
        st.pyplot(fig)

    # --- Emotion over time ---
    st.subheader("Emotion Across Conversation Turns")
    turn_sequence = [
        {"turn_number": t["turn_number"], "emotion": t.get("emotion","neutral")}
        for t in all_turns if t.get("emotion")
    ]
    fig = chart_emotion_over_time(turn_sequence)
    st.pyplot(fig)

    # --- Latency comparison ---
    st.subheader("Response Latency by System")
    latencies_a = [t["latency_ms"] for t in all_turns
                   if t.get("response_system") == "normal" and t.get("latency_ms",0) > 0]
    latencies_b = [t["latency_ms"] for t in all_turns
                   if t.get("response_system") == "emotion_aware" and t.get("latency_ms",0) > 0]
    if latencies_a or latencies_b:
        fig = chart_latency_comparison(latencies_a, latencies_b)
        st.pyplot(fig)
    else:
        st.caption("No latency data yet. Use the comparison page to generate side-by-side responses.")

    st.divider()

    # --- Student ratings ---
    st.subheader("Student Evaluation Ratings")
    stats = get_summary_stats(include_demo=False)
    if stats["total_ratings"] == 0:
        st.info("No student ratings submitted yet. Use the Student Evaluation page to add ratings.")
    else:
        fig = chart_ratings_comparison(stats)
        st.pyplot(fig)

        col1, col2 = st.columns(2)
        col1.metric("Total Ratings", stats["total_ratings"])
        with col2:
            n = stats["normal"]["avg_overall"]
            a = stats["emotion_aware"]["avg_overall"]
            diff = round(a - n, 2)
            col2.metric("Emotion-Aware vs Normal (overall avg)", f"{diff:+.2f} pts", delta=f"{diff:+.2f}")

    st.divider()
    st.caption(
        "⚠️ **Data transparency note:** All analytics shown here are based on actual session usage "
        "and real student ratings. No data is fabricated or simulated. "
        "Session data resets when the browser is closed unless persistent storage is configured."
    )
