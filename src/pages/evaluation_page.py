"""
Streamlit Page — Student Evaluator Ratings (Phase 9)
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.evaluation.rating_store import init_db, save_rating, get_all_ratings, get_summary_stats, get_feedback_comments
import pandas as pd


def render():
    init_db()
    st.title("⭐ Student Evaluator Ratings")
    st.caption("Rate the chatbot responses to help evaluate the system.")

    tab1, tab2 = st.tabs(["📝 Submit Rating", "📊 View Results"])

    with tab1:
        st.subheader("Rate a Response")
        st.info(
            "After chatting with the assistant or using the comparison feature, "
            "come here to rate the response quality."
        )

        with st.form("rating_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                system = st.selectbox("Which system are you rating?",
                                      ["emotion_aware", "normal"],
                                      format_func=lambda x: "✨ Emotion-Aware Chatbot" if x == "emotion_aware" else "🤖 Normal Chatbot")
            with col2:
                emotion = st.selectbox("Detected emotion (if known)",
                                       ["neutral","joy","sadness","anger","fear","surprise","disgust","(unknown)"])

            user_message = st.text_area("Your message (optional — paste what you asked)", height=60)
            response_text = st.text_area("The response you are rating (optional — paste it here)", height=100)

            st.markdown("**Rate the response (1 = Poor, 5 = Excellent):**")
            c1, c2, c3, c4 = st.columns(4)
            empathy    = c1.slider("🤝 Empathy",    1, 5, 3, key="emp")
            relevance  = c2.slider("🎯 Relevance",  1, 5, 3, key="rel")
            usefulness = c3.slider("💡 Usefulness", 1, 5, 3, key="use")
            tone       = c4.slider("🗣️ Tone",        1, 5, 3, key="ton")

            feedback = st.text_area("Optional written feedback", height=80, placeholder="What worked well? What could be better?")

            submitted = st.form_submit_button("Submit Rating", use_container_width=True, type="primary")

        if submitted:
            # Get session_id from active conversation if available
            session_id = st.session_state.get("chat_manager", {})
            if hasattr(session_id, "session_id"):
                session_id = session_id.session_id
            else:
                session_id = "manual_eval"

            try:
                eval_id = save_rating(
                    session_id        = session_id,
                    system            = system,
                    empathy_rating    = empathy,
                    relevance_rating  = relevance,
                    usefulness_rating = usefulness,
                    tone_rating       = tone,
                    user_message      = user_message,
                    response_text     = response_text,
                    emotion           = emotion if emotion != "(unknown)" else "neutral",
                    feedback_text     = feedback,
                    is_demo           = False,
                )
                st.success(f"✅ Rating submitted! (ID: `{eval_id[:20]}...`)")
            except Exception as e:
                st.error(f"Failed to save rating: {e}")

    with tab2:
        st.subheader("Evaluation Results")
        stats = get_summary_stats(include_demo=False)

        if stats["total_ratings"] == 0:
            st.info("No real student ratings yet. Submit your first rating in the 'Submit Rating' tab.")
            st.caption("Demo ratings are excluded from this view.")
        else:
            st.metric("Total Ratings", stats["total_ratings"])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🤖 Normal Chatbot**")
                n = stats["normal"]
                if n["count"] > 0:
                    st.metric("Ratings",    n["count"])
                    st.metric("Empathy",    n["empathy"])
                    st.metric("Relevance",  n["relevance"])
                    st.metric("Usefulness", n["usefulness"])
                    st.metric("Tone",       n["tone"])
                    st.metric("Overall Avg", n["avg_overall"])
                else:
                    st.caption("No ratings yet.")

            with col2:
                st.markdown("**✨ Emotion-Aware Chatbot**")
                a = stats["emotion_aware"]
                if a["count"] > 0:
                    st.metric("Ratings",    a["count"])
                    st.metric("Empathy",    a["empathy"])
                    st.metric("Relevance",  a["relevance"])
                    st.metric("Usefulness", a["usefulness"])
                    st.metric("Tone",       a["tone"])
                    st.metric("Overall Avg", a["avg_overall"])
                else:
                    st.caption("No ratings yet.")

        st.divider()
        st.subheader("Written Feedback")
        comments = get_feedback_comments(limit=20, include_demo=False)
        if comments:
            for c in comments:
                st.markdown(f"**{c['system']} | {c['emotion']}** — _{c['timestamp'][:10]}_")
                st.caption(c["feedback_text"])
                st.divider()
        else:
            st.caption("No feedback comments yet.")

        st.divider()
        st.subheader("Raw Data")
        df = get_all_ratings(include_demo=False)
        if not df.empty:
            st.dataframe(
                df[["timestamp","system","emotion","empathy_rating","relevance_rating","usefulness_rating","tone_rating","feedback_text"]],
                use_container_width=True
            )
        else:
            st.caption("No data yet.")
