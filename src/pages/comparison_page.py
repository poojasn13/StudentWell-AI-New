"""
Streamlit Page — Normal vs Emotion-Aware Comparison (Phase 8)
"""

import time
import streamlit as st
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.safety.safety_checker import check_safety, SafetyLevel
from src.generation.llm_response import generate_both_responses, check_ollama_available
from src.conversation.conversation_manager import ComparisonSessionManager

EMOTION_EMOJI = {
    "neutral":"😐","joy":"😊","sadness":"😢",
    "anger":"😠","fear":"😰","surprise":"😲","disgust":"😒",
}


def _get_cmp_manager() -> ComparisonSessionManager:
    if "cmp_manager" not in st.session_state:
        st.session_state["cmp_manager"] = ComparisonSessionManager()
    return st.session_state["cmp_manager"]


def _get_predictor():
    if "emotion_predictor" not in st.session_state:
        try:
            from src.model.predict_emotion import EmotionPredictor
            st.session_state["emotion_predictor"] = EmotionPredictor()
        except FileNotFoundError:
            st.session_state["emotion_predictor"] = None
    return st.session_state["emotion_predictor"]


def render():
    st.title("⚖️ Normal vs Emotion-Aware Comparison")
    st.caption(
        "Same question — two different response styles. "
        "System A has no emotion detection. System B adapts to your emotional state."
    )

    available, msg = check_ollama_available()
    if not available:
        st.error(f"⚠️ {msg}")
        st.stop()

    predictor = _get_predictor()
    if predictor is None:
        st.info("ℹ️ Emotion classifier not trained yet — comparison will use neutral emotion fallback.")

    cmp_mgr = _get_cmp_manager()

    with st.sidebar:
        st.markdown("### ℹ️ System Descriptions")
        st.markdown("**System A — Normal Chatbot**")
        st.caption("Qwen2.5:3b with a neutral system prompt. No emotion detection.")
        st.markdown("**System B — Emotion-Aware**")
        st.caption("DistilBERT detects emotion → Qwen2.5:3b adapts tone, empathy, and structure.")
        st.divider()
        if st.button("🔄 Reset Comparison", use_container_width=True):
            cmp_mgr.reset()
            st.session_state["cmp_history"] = []
            st.rerun()

    if "cmp_history" not in st.session_state:
        st.session_state["cmp_history"] = []

    # Show previous turns
    for entry in st.session_state["cmp_history"]:
        st.markdown(f"**You:** {entry['user']}")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🤖 System A — Normal**")
            st.info(entry["response_a"])
            st.caption(f"⏱ {entry['latency_a']:.0f}ms")
        with col_b:
            em = entry.get("emotion", "neutral")
            st.markdown(f"**✨ System B — Emotion-Aware** {EMOTION_EMOJI.get(em,'')} `{em}`")
            st.success(entry["response_b"])
            st.caption(f"⏱ {entry['latency_b']:.0f}ms | conf: {entry.get('confidence',0):.0%}")
        st.divider()

    user_input = st.chat_input("Type a message to compare both systems...")
    if not user_input:
        return

    safety = check_safety(user_input)
    if safety.triggered:
        if safety.level == SafetyLevel.CRISIS:
            st.error(safety.response_text)
        else:
            st.warning(safety.response_text)
        return

    # Emotion detection
    context_b = cmp_mgr.system_b.get_classifier_context()
    if predictor:
        pred   = predictor.predict(user_input, context_b)
        emotion, confidence, low_conf = pred["emotion"], pred["confidence"], pred["low_confidence"]
    else:
        emotion, confidence, low_conf = "neutral", 0.0, True

    # Generate both responses
    with st.spinner("Generating both responses..."):
        hist_a = cmp_mgr.system_a.get_ollama_history()
        hist_b = cmp_mgr.system_b.get_ollama_history()
        result_a, result_b = generate_both_responses(
            user_message            = user_input,
            conversation_history_a  = hist_a,
            conversation_history_b  = hist_b,
            emotion                 = emotion,
            confidence              = confidence,
            low_confidence          = low_conf,
        )

    # Update managers
    t_a = cmp_mgr.system_a.add_user_message(user_input)
    cmp_mgr.system_a.update_turn(t_a, assistant_message=result_a.text, emotion="neutral",
                                  response_system="normal", latency_ms=result_a.latency_ms)
    t_b = cmp_mgr.system_b.add_user_message(user_input)
    cmp_mgr.system_b.update_turn(t_b, assistant_message=result_b.text, emotion=emotion,
                                  confidence=confidence, response_system="emotion_aware",
                                  latency_ms=result_b.latency_ms)

    # Display
    st.markdown(f"**You:** {user_input}")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**🤖 System A — Normal**")
        if result_a.success:
            st.info(result_a.text)
            st.caption(f"⏱ {result_a.latency_ms:.0f}ms")
        else:
            st.error(f"Error: {result_a.error}")
    with col_b:
        em = emotion
        st.markdown(f"**✨ System B — Emotion-Aware** {EMOTION_EMOJI.get(em,'')} `{em}`")
        if result_b.success:
            st.success(result_b.text)
            st.caption(f"⏱ {result_b.latency_ms:.0f}ms | conf: {confidence:.0%}")
        else:
            st.error(f"Error: {result_b.error}")

    # Store in history
    st.session_state["cmp_history"].append({
        "user":       user_input,
        "response_a": result_a.text if result_a.success else f"Error: {result_a.error}",
        "response_b": result_b.text if result_b.success else f"Error: {result_b.error}",
        "latency_a":  result_a.latency_ms,
        "latency_b":  result_b.latency_ms,
        "emotion":    emotion,
        "confidence": confidence,
    })
    st.divider()
