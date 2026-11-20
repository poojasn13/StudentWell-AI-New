"""
Chat page — every message gets TWO responses automatically:
  🤖 Normal      — short, plain, factual, bullet-point style
  ✨ Emotion-Aware — warm, empathetic, lengthy, calming, emotion-adapted
No toggle needed. Both always shown side by side.
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.safety.safety_checker           import check_safety, SafetyLevel
from src.generation.llm_response         import (generate_emotion_aware_response,
                                                  generate_normal_response,
                                                  check_ollama_available)
from src.conversation.conversation_manager import ConversationManager
from src.evaluation.rating_store          import init_db, save_rating, get_all_ratings, get_summary_stats

EMOTION_EMOJI = {"neutral":"😐","joy":"😊","sadness":"😢",
                 "anger":"😠","fear":"😰","surprise":"😲","disgust":"😒"}
EMOTION_COLOR = {"neutral":"#64748b","joy":"#eab308","sadness":"#3b82f6",
                 "anger":"#ef4444","fear":"#8b5cf6","surprise":"#06b6d4","disgust":"#22c55e"}
EMOTION_BG    = {"neutral":"#1e293b","joy":"#1c1500","sadness":"#0f1f3d",
                 "anger":"#2d0a0a","fear":"#1a0d2e","surprise":"#062030","disgust":"#052010"}
EMOTION_DESC  = {
    "neutral":  "Calm & informational. Gets a direct, structured answer.",
    "joy":      "Happy or excited. Gets a warm, celebratory reply that channels energy forward.",
    "sadness":  "Sad or discouraged. Gets empathy first, then gentle practical steps.",
    "anger":    "Frustrated or angry. Gets calm, non-judgmental, structured de-escalation.",
    "fear":     "Anxious or nervous. Gets reassurance + step-by-step confidence building.",
    "surprise": "Something unexpected. Gets a curious, responsive, explanatory reply.",
    "disgust":  "Strong negative reaction. Gets a respectful, solution-oriented response.",
}


# ── State helpers ─────────────────────────────────────────────────────────────

def _get_manager() -> ConversationManager:
    if st.session_state.get("chat_manager") is None:
        st.session_state["chat_manager"] = ConversationManager()
    return st.session_state["chat_manager"]

def _get_predictor():
    if "emotion_predictor" not in st.session_state:
        try:
            from src.model.predict_emotion import EmotionPredictor
            st.session_state["emotion_predictor"] = EmotionPredictor()
        except FileNotFoundError:
            st.session_state["emotion_predictor"] = None
    return st.session_state["emotion_predictor"]

def _get_messages() -> list:
    if "current_messages" not in st.session_state:
        st.session_state["current_messages"] = []
    return st.session_state["current_messages"]

def _save_conversation(manager, messages):
    if not messages:
        return
    convs = st.session_state.setdefault("conversations", [])
    idx   = st.session_state.get("active_conv_index")
    # Safely extract the first user text regardless of message type
    first = "Chat"
    for m in messages:
        if m.get("type") == "dual":
            first = m.get("user", "Chat")
            break
        elif m.get("type") == "safety":
            first = m.get("user", "Chat")
            break
        elif m.get("role") == "user":
            first = m.get("content", "Chat")
            break
    title = str(first)[:40] + ("…" if len(str(first)) > 40 else "")
    entry = {
        "title":        title,
        "messages":     list(messages),
        "manager_dict": manager.to_dict(),
        "turn_count":   manager.turn_count,
        "emotion_dist": manager.get_emotion_distribution(),
        "avg_conf":     manager.average_confidence,
    }
    if idx is not None and idx < len(convs):
        convs[idx] = entry
    else:
        convs.append(entry)
        st.session_state["active_conv_index"] = len(convs) - 1


# ── Emotion legend ────────────────────────────────────────────────────────────

def _emotion_legend():
    with st.expander("🎭 7 Emotions — click to expand", expanded=False):
        st.markdown("""<div style='font-size:11px;color:#4b5563;margin-bottom:8px;'>
        DistilBERT classifies every message into one of these 7 emotions.
        The detected emotion changes how the Emotion-Aware response is written.
        </div>""", unsafe_allow_html=True)
        for em, desc in EMOTION_DESC.items():
            color = EMOTION_COLOR[em]
            emoji = EMOTION_EMOJI[em]
            bg    = EMOTION_BG[em]
            st.markdown(f"""
            <div style='display:flex;align-items:flex-start;gap:10px;margin-bottom:7px;
                        background:{bg};border-left:3px solid {color};border-radius:8px;
                        padding:8px 11px;'>
              <span style='font-size:17px;line-height:1.3;'>{emoji}</span>
              <div>
                <div style='font-size:12px;font-weight:700;color:{color};
                            text-transform:capitalize;margin-bottom:2px;'>{em}</div>
                <div style='font-size:11px;color:#94a3b8;line-height:1.5;'>{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)


# ── Analytics panel ───────────────────────────────────────────────────────────

def _analytics_panel(manager: ConversationManager):

    # Session metrics
    st.markdown("""<div style='background:#161b27;border:1px solid #2a3449;border-radius:12px;
                padding:14px 16px;margin-bottom:10px;'>
      <div style='font-size:10px;font-weight:700;color:#4b5563;
                  letter-spacing:.08em;margin-bottom:10px;'>📈 SESSION ANALYTICS</div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Messages",  manager.turn_count)
    c2.metric("Avg Conf.", f"{manager.average_confidence:.0%}" if manager.average_confidence else "—")
    st.markdown("</div>", unsafe_allow_html=True)

    # Emotion distribution
    dist = manager.get_emotion_distribution()
    if dist:
        total = sum(dist.values())
        st.markdown("""<div style='background:#161b27;border:1px solid #2a3449;border-radius:12px;
                    padding:14px 16px;margin-bottom:10px;'>
          <div style='font-size:10px;font-weight:700;color:#4b5563;
                      letter-spacing:.08em;margin-bottom:10px;'>🎭 EMOTIONS IN THIS CHAT</div>
        """, unsafe_allow_html=True)
        for em, cnt in sorted(dist.items(), key=lambda x: -x[1]):
            pct   = int(cnt / total * 100)
            color = EMOTION_COLOR.get(em, "#64748b")
            emoji = EMOTION_EMOJI.get(em, "")
            st.markdown(f"""
            <div style='margin-bottom:7px;'>
              <div style='display:flex;justify-content:space-between;margin-bottom:3px;'>
                <span style='font-size:12px;color:#cbd5e1;'>{emoji} {em}</span>
                <span style='font-size:11px;color:#4b5563;'>{cnt} ({pct}%)</span>
              </div>
              <div style='background:#1e293b;border-radius:4px;height:5px;'>
                <div style='background:{color};width:{max(pct,2)}%;height:5px;border-radius:4px;'></div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Turn timeline
    turns = manager.session.turns
    if turns:
        st.markdown("""<div style='background:#161b27;border:1px solid #2a3449;border-radius:12px;
                    padding:14px 16px;margin-bottom:10px;'>
          <div style='font-size:10px;font-weight:700;color:#4b5563;
                      letter-spacing:.08em;margin-bottom:10px;'>🔄 TURN TIMELINE</div>
        """, unsafe_allow_html=True)
        for t in turns[-6:]:
            em      = t.emotion
            color   = EMOTION_COLOR.get(em, "#64748b")
            bg      = EMOTION_BG.get(em, "#1e293b")
            emoji   = EMOTION_EMOJI.get(em, "")
            conf    = f"{t.confidence:.0%}" if t.confidence else ""
            preview = (t.user_message[:25] + "…") if len(t.user_message) > 25 else t.user_message
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:5px;margin-bottom:5px;
                        background:{bg};border-radius:7px;padding:5px 8px;
                        border-left:3px solid {color};'>
              <span style='font-size:10px;color:#374151;min-width:16px;'>#{t.turn_number}</span>
              <span style='font-size:11px;color:#94a3b8;flex:1;overflow:hidden;
                           text-overflow:ellipsis;white-space:nowrap;'>{preview}</span>
              <span style='font-size:10px;background:{color}22;color:{color};
                           border-radius:10px;padding:1px 6px;white-space:nowrap;'>{emoji} {em}</span>
              <span style='font-size:10px;color:#374151;'>{conf}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Safety
    sc = manager.safety_trigger_count
    if sc > 0:
        st.markdown(f"""<div style='background:#2d0a0a;border:1px solid #7f1d1d;border-radius:10px;
                    padding:10px 14px;margin-bottom:10px;'>
          <div style='font-size:12px;color:#fca5a5;'>🛡️ Safety triggers: <b>{sc}</b></div>
        </div>""", unsafe_allow_html=True)

    # Ratings from DB
    _ratings_summary()

    # Clear button
    if st.button("🗑️ Clear This Chat", use_container_width=True, key="clear_chat_btn"):
        st.session_state["chat_manager"]      = None
        st.session_state["current_messages"]  = []
        st.session_state["active_conv_index"] = None
        st.rerun()


def _ratings_summary():
    try:
        stats = get_summary_stats(include_demo=False)
        total = stats["total_ratings"]
    except Exception:
        return

    st.markdown("""<div style='background:#161b27;border:1px solid #2a3449;border-radius:12px;
                padding:14px 16px;margin-bottom:10px;'>
      <div style='font-size:10px;font-weight:700;color:#4b5563;
                  letter-spacing:.08em;margin-bottom:10px;'>⭐ RATINGS SUBMITTED</div>
    """, unsafe_allow_html=True)

    if total == 0:
        st.markdown("""<div style='font-size:11.5px;color:#374151;'>
          No ratings yet — rate a response below ↓
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div style='font-size:12px;color:#94a3b8;margin-bottom:10px;'>
          Total: <b style='color:#e2e8f0;'>{total}</b> ratings submitted
        </div>""", unsafe_allow_html=True)

        for key, label, color, bg in [
            ("emotion_aware", "✨ Emotion-Aware", "#22c55e", "#052010"),
            ("normal",        "🤖 Normal",         "#3b82f6", "#0a1628"),
        ]:
            s = stats[key]
            if s["count"] == 0:
                continue
            st.markdown(f"""
            <div style='background:{bg};border:1px solid {color}44;border-radius:9px;
                        padding:9px 12px;margin-bottom:7px;'>
              <div style='font-size:11.5px;font-weight:700;color:{color};margin-bottom:6px;'>{label}</div>
              <div style='display:grid;grid-template-columns:1fr 1fr;gap:3px;'>
                <div style='font-size:10.5px;color:#94a3b8;'>🤝 Empathy</div>
                <div style='font-size:10.5px;color:#e2e8f0;text-align:right;'>{s["empathy"]}/5</div>
                <div style='font-size:10.5px;color:#94a3b8;'>🎯 Relevance</div>
                <div style='font-size:10.5px;color:#e2e8f0;text-align:right;'>{s["relevance"]}/5</div>
                <div style='font-size:10.5px;color:#94a3b8;'>💡 Usefulness</div>
                <div style='font-size:10.5px;color:#e2e8f0;text-align:right;'>{s["usefulness"]}/5</div>
                <div style='font-size:10.5px;color:#94a3b8;'>🗣 Tone</div>
                <div style='font-size:10.5px;color:#e2e8f0;text-align:right;'>{s["tone"]}/5</div>
              </div>
              <div style='margin-top:5px;padding-top:5px;border-top:1px solid {color}22;
                          font-size:11px;color:{color};font-weight:600;'>
                Overall: {s["avg_overall"]}/5 &nbsp;·&nbsp; {s["count"]} ratings
              </div>
            </div>""", unsafe_allow_html=True)

        # Last 3 comments
        try:
            df = get_all_ratings(include_demo=False)
            fb = df[df["feedback_text"].str.strip() != ""][
                ["system","emotion","feedback_text"]].tail(3)
            if not fb.empty:
                st.markdown("""<div style='font-size:10px;font-weight:700;color:#374151;
                                letter-spacing:.06em;margin:8px 0 4px;'>RECENT COMMENTS</div>""",
                            unsafe_allow_html=True)
                for _, row in fb.iterrows():
                    sc = "#22c55e" if row["system"] == "emotion_aware" else "#3b82f6"
                    si = "✨" if row["system"] == "emotion_aware" else "🤖"
                    st.markdown(f"""
                    <div style='background:#0d1117;border-radius:7px;padding:7px 10px;
                                margin-bottom:5px;border-left:2px solid {sc};'>
                      <div style='font-size:10px;color:{sc};margin-bottom:2px;'>
                        {si} {row["system"]} · {row["emotion"]}
                      </div>
                      <div style='font-size:11px;color:#94a3b8;font-style:italic;'>
                        "{str(row["feedback_text"])[:80]}{"…" if len(str(row["feedback_text"]))>80 else ""}"
                      </div>
                    </div>""", unsafe_allow_html=True)
        except Exception:
            pass

    st.markdown("</div>", unsafe_allow_html=True)


# ── Rating widget ─────────────────────────────────────────────────────────────

def _rating_widget(manager: ConversationManager, messages: list):
    has_reply = any(m.get("type") == "dual" for m in messages)
    if not has_reply:
        return

    with st.expander("⭐ Rate the last response", expanded=False):
        st.caption("Rate the quality of the last response (1 = poor, 5 = excellent).")
        c1, c2, c3, c4 = st.columns(4)
        emp  = c1.select_slider("🤝 Empathy",    [1,2,3,4,5], value=3, key="r_emp")
        rel  = c2.select_slider("🎯 Relevance",  [1,2,3,4,5], value=3, key="r_rel")
        use  = c3.select_slider("💡 Usefulness", [1,2,3,4,5], value=3, key="r_use")
        tone = c4.select_slider("🗣 Tone",        [1,2,3,4,5], value=3, key="r_tone")
        fb   = st.text_input("Comments (optional)", key="r_feedback",
                             placeholder="What worked well or could be better?")
        system = st.radio("Which response are you rating?",
                          ["emotion_aware","normal"],
                          format_func=lambda x: "✨ Emotion-Aware" if x=="emotion_aware" else "🤖 Normal",
                          horizontal=True, key="r_system")

        if st.button("Submit Rating ✓", key="submit_rating_btn", type="primary",
                     use_container_width=True):
            last = manager.session.turns
            emotion = last[-1].emotion if last else "neutral"
            last_dual = next((m for m in reversed(messages) if m.get("type") == "dual"), {})
            last_q    = last_dual.get("user", "")
            last_r    = last_dual.get("resp_aware","") if system=="emotion_aware" else last_dual.get("resp_normal","")
            try:
                init_db()
                eid = save_rating(
                    session_id=manager.session_id, system=system,
                    empathy_rating=emp, relevance_rating=rel,
                    usefulness_rating=use, tone_rating=tone,
                    turn_number=manager.turn_count,
                    user_message=last_q, response_text=last_r[:500],
                    emotion=emotion, feedback_text=fb, is_demo=False,
                )
                st.success(f"✅ Rating saved!")
                st.rerun()
            except Exception as e:
                st.error(f"Could not save: {e}")


# ── Dual response renderer ────────────────────────────────────────────────────

def _render_dual(msg: dict, show_debug: bool):
    em    = msg.get("emotion", "neutral")
    color = EMOTION_COLOR.get(em, "#64748b")
    emoji = EMOTION_EMOJI.get(em, "")

    # Student message bubble
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(msg["user"])

    # Detected emotion tag
    if show_debug:
        conf = msg.get("confidence", 0)
        lc   = " ⚠️ low conf" if msg.get("low_confidence") else ""
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:8px;margin:4px 0 10px 44px;'>
          <span style='font-size:11px;color:#4b5563;'>Detected emotion:</span>
          <span style='background:{color}22;color:{color};border:1px solid {color}44;
                       border-radius:20px;padding:2px 10px;font-size:11px;font-weight:600;'>
            {emoji} {em}{lc}
          </span>
          <span style='font-size:11px;color:#374151;'>conf: {conf:.0%}</span>
        </div>""", unsafe_allow_html=True)

    # Two response columns
    col_n, col_e = st.columns(2, gap="small")

    # ── Normal response (left) ────────────────────────────────────────────────
    with col_n:
        lat_n = msg.get("lat_normal", 0)
        st.markdown(f"""
        <div style='background:#0d1525;border:1.5px solid #1e3a5f;border-radius:12px;
                    padding:14px 16px;min-height:120px;'>
          <div style='display:flex;align-items:center;gap:8px;margin-bottom:10px;
                      padding-bottom:8px;border-bottom:1px solid #1e3a5f;'>
            <span style='font-size:18px;'>🤖</span>
            <div>
              <div style='font-size:12px;font-weight:700;color:#60a5fa;'>Normal Response</div>
              <div style='font-size:10px;color:#374151;'>Plain · factual · no emotion awareness</div>
            </div>
          </div>
          <div style='font-size:13.5px;color:#94a3b8;line-height:1.65;'>{msg["resp_normal"]}</div>
          <div style='font-size:10px;color:#2a3449;margin-top:10px;text-align:right;'>⏱ {lat_n:.0f}ms</div>
        </div>""", unsafe_allow_html=True)

    # ── Emotion-aware response (right) ────────────────────────────────────────
    with col_e:
        lat_e  = msg.get("lat_aware", 0)
        em_bg  = EMOTION_BG.get(em, "#052010")
        st.markdown(f"""
        <div style='background:{em_bg};border:1.5px solid {color};border-radius:12px;
                    padding:14px 16px;min-height:120px;'>
          <div style='display:flex;align-items:center;gap:8px;margin-bottom:10px;
                      padding-bottom:8px;border-bottom:1px solid {color}44;'>
            <span style='font-size:18px;'>✨</span>
            <div>
              <div style='font-size:12px;font-weight:700;color:{color};'>Emotion-Aware Response</div>
              <div style='font-size:10px;color:#374151;'>Empathetic · calming · adapted to {emoji} {em}</div>
            </div>
          </div>
          <div style='font-size:13.5px;color:#e2e8f0;line-height:1.75;'>{msg["resp_aware"]}</div>
          <div style='font-size:10px;color:#374151;margin-top:10px;text-align:right;'>⏱ {lat_e:.0f}ms</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    init_db()

    available, ollama_msg = check_ollama_available()
    if not available:
        st.error(f"⚠️ Ollama not running — {ollama_msg}")
        st.info("Open a terminal and run: `ollama serve`")
        st.stop()

    predictor = _get_predictor()
    manager   = _get_manager()
    messages  = _get_messages()

    chat_col, right_col = st.columns([3, 1], gap="medium")

    # ── RIGHT PANEL ───────────────────────────────────────────────────────────
    with right_col:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # Classifier chip
        if predictor is None:
            st.markdown("""<div style='background:#1c1500;border:1px solid #ca8a04;
                border-radius:9px;padding:9px 12px;margin-bottom:4px;
                font-size:11.5px;color:#fbbf24;'>
                ⚠️ Classifier not trained<br>
                <span style='font-size:10px;color:#6b7280;'>Run: python run_pipeline.py</span>
              </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div style='background:#052010;border:1px solid #22c55e;
                border-radius:9px;padding:9px 12px;margin-bottom:4px;
                font-size:11.5px;color:#4ade80;'>
                ✅ DistilBERT active
              </div>""", unsafe_allow_html=True)
            _emotion_legend()

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # Show emotion details toggle only
        show_debug = st.toggle("🔍 Show emotion details", value=True, key="debug_toggle")

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        _analytics_panel(manager)
        _rating_widget(manager, messages)

    # ── CHAT AREA ─────────────────────────────────────────────────────────────
    with chat_col:
        # Header
        st.markdown("""
        <div style='padding-bottom:10px;border-bottom:1px solid #2a3449;margin-bottom:14px;'>
          <div style='font-size:19px;font-weight:700;color:#e2e8f0;'>💬 StudentWell AI</div>
          <div style='font-size:12px;color:#6b7280;margin-top:3px;'>
            Every response shows <b style='color:#60a5fa;'>🤖 Normal</b> and
            <b style='color:#22c55e;'>✨ Emotion-Aware</b> side by side —
            see the difference emotion detection makes.
          </div>
        </div>""", unsafe_allow_html=True)

        # Welcome screen
        if not messages:
            st.markdown("""
            <div style='text-align:center;padding:28px 16px;'>
              <div style='font-size:36px;margin-bottom:10px;'>🎓</div>
              <div style='font-size:15px;font-weight:600;color:#94a3b8;margin-bottom:6px;'>
                How can I help you today?
              </div>
              <div style='font-size:12.5px;color:#374151;max-width:380px;
                          margin:0 auto 20px;line-height:1.6;'>
                Type anything. You'll see a plain factual answer on the left and a warm,
                emotion-adapted response on the right — both generated from the same question.
              </div>
            </div>""", unsafe_allow_html=True)
            suggestions = [
                "I have an exam tomorrow 😰",
                "I failed my assignment 😢",
                "I got top marks today! 😊",
                "How should I manage my time?",
            ]
            c1, c2 = st.columns(2)
            for i, s in enumerate(suggestions):
                if [c1, c2][i % 2].button(s, key=f"sug_{i}", use_container_width=True):
                    st.session_state["_prefill"] = s
                    st.rerun()

        # Render existing messages
        for msg in messages:
            if msg.get("type") == "dual":
                _render_dual(msg, show_debug)
            elif msg.get("type") == "safety":
                with st.chat_message("user", avatar="🧑‍🎓"):
                    st.markdown(msg["user"])
                with st.chat_message("assistant", avatar="🛡️"):
                    if msg.get("level") == "crisis":
                        st.error(msg["response"])
                    else:
                        st.warning(msg["response"])

        # ── Input ──────────────────────────────────────────────────────────────
        prefill    = st.session_state.pop("_prefill", "")
        user_input = st.chat_input(
            "Ask anything about studies, exams, assignments…",
            key="chat_input_main",
        ) or prefill

        if not user_input:
            return

        # Safety check
        safety = check_safety(user_input)
        if safety.triggered:
            turn_num = manager.add_user_message(user_input)
            manager.update_turn(turn_num, assistant_message=safety.response_text,
                                safety_triggered=True, safety_level=safety.level.value)
            messages.append({
                "type":     "safety",
                "user":     user_input,
                "response": safety.response_text,
                "level":    "crisis" if safety.level == SafetyLevel.CRISIS else "concern",
            })
            _save_conversation(manager, messages)
            st.rerun()
            return

        # Emotion prediction
        context = manager.get_classifier_context()
        if predictor is not None:
            pred           = predictor.predict(user_input, context)
            emotion        = pred["emotion"]
            confidence     = pred["confidence"]
            low_confidence = pred["low_confidence"]
        else:
            emotion, confidence, low_confidence = "neutral", 0.0, True

        # Generate BOTH responses in parallel (sequential for Ollama)
        with st.spinner("Generating both responses…"):
            hist          = manager.get_ollama_history()
            result_normal = generate_normal_response(user_input, hist)
            result_aware  = generate_emotion_aware_response(
                user_input, hist, emotion, confidence, low_confidence
            )

        # Save to manager (emotion-aware is the canonical turn)
        turn_num = manager.add_user_message(user_input)
        manager.update_turn(
            turn_num,
            assistant_message = result_aware.text if result_aware.success else "",
            emotion           = emotion,
            confidence        = confidence,
            low_confidence    = low_confidence,
            response_system   = "emotion_aware",
            latency_ms        = result_aware.latency_ms,
        )

        messages.append({
            "type":          "dual",
            "user":          user_input,
            "resp_normal":   result_normal.text if result_normal.success else f"⚠️ Error: {result_normal.error}",
            "resp_aware":    result_aware.text  if result_aware.success  else f"⚠️ Error: {result_aware.error}",
            "lat_normal":    result_normal.latency_ms,
            "lat_aware":     result_aware.latency_ms,
            "emotion":       emotion,
            "confidence":    confidence,
            "low_confidence": low_confidence,
        })

        _save_conversation(manager, messages)
        st.rerun()
