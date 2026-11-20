"""
StudentWell AI — Emotion-Aware Student Support Assistant
Run:  streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import streamlit as st

st.set_page_config(
    page_title="StudentWell AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f1117; color: #e8eaf0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
section[data-testid="stSidebar"] {
    background-color: #161b27 !important;
    border-right: 1px solid #2a3449;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] * { color: #c8cdd8 !important; }

/* Nav radio */
[data-testid="stSidebar"] .stRadio > div { gap: 2px; }
[data-testid="stSidebar"] .stRadio label {
    padding: 9px 14px !important; border-radius: 8px !important;
    font-size: 13px !important; font-weight: 500 !important;
    cursor: pointer; transition: background 0.15s; display: block; width: 100%;
}
[data-testid="stSidebar"] .stRadio label:hover { background: #1f2535 !important; }

/* History items */
.hist-item {
    padding: 9px 12px; margin: 2px 0; border-radius: 8px;
    font-size: 12.5px; color: #9ca3af; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
    border: 1px solid #1e2435; background: #12161f;
    transition: all 0.15s;
}
.hist-label {
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em;
    color: #374151; padding: 10px 4px 4px; font-weight: 600;
}
/* Metrics */
[data-testid="stMetric"] {
    background: #1a1f2e; border-radius: 10px;
    padding: 12px 16px; border: 1px solid #2a3449;
}
[data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 11px !important; }
[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 20px !important; font-weight: 700 !important; }
/* Buttons */
.stButton > button {
    background: #1e3a5f !important; color: #93c5fd !important;
    border: 1px solid #2563eb !important; border-radius: 8px !important;
    font-size: 12.5px !important; transition: all 0.15s;
}
.stButton > button:hover { background: #2563eb !important; color: #fff !important; }
/* Chat input */
[data-testid="stChatInput"] textarea {
    background: #1a1f2e !important; border: 1px solid #2a3449 !important;
    border-radius: 12px !important; color: #e2e8f0 !important; font-size: 14px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #2563eb !important; box-shadow: 0 0 0 2px rgba(37,99,235,0.2) !important;
}
/* Chat messages */
[data-testid="stChatMessage"] {
    padding: 10px 16px !important; border-radius: 14px !important;
    margin-bottom: 8px !important; border: 1px solid #1e2435;
}
hr { border-color: #2a3449 !important; margin: 8px 0 !important; }
h1 { color: #e2e8f0 !important; font-weight: 700 !important; font-size: 22px !important; }
h2 { color: #cbd5e1 !important; font-weight: 600 !important; font-size: 18px !important; }
h3 { color: #94a3b8 !important; font-weight: 600 !important; font-size: 15px !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
/* Do NOT hide header — it contains the sidebar toggle button */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #2a3449; border-radius: 4px; }
[data-testid="stMetric"] { background: #1a1f2e; border-radius: 10px; padding: 12px; border: 1px solid #2a3449; }
</style>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────────────────────
if "conversations" not in st.session_state:
    st.session_state["conversations"] = []
if "active_conv_index" not in st.session_state:
    st.session_state["active_conv_index"] = None

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:18px 12px 12px; border-bottom:1px solid #2a3449; margin-bottom:8px;'>
        <div style='font-size:20px; font-weight:700; color:#e2e8f0;'>🎓 StudentWell AI</div>
        <div style='font-size:11px; color:#6b7280; margin-top:2px;'>Emotion-Aware Student Support</div>
    </div>
    """, unsafe_allow_html=True)

    PAGES = {
        "💬  Chat":               "chat",
        "🎯  Model Performance":  "model_performance",
        "ℹ️  About":               "about",
    }
    st.markdown('<div class="hist-label">Navigation</div>', unsafe_allow_html=True)
    page_label = st.radio("nav", list(PAGES.keys()), label_visibility="collapsed", key="nav_radio")
    page_key   = PAGES[page_label]

    # Conversation history — only on Chat page
    if page_key == "chat":
        st.divider()
        st.markdown('<div class="hist-label">Conversation History</div>', unsafe_allow_html=True)

        col_new, _ = st.columns([3, 1])
        with col_new:
            if st.button("＋  New Chat", use_container_width=True, key="new_chat_btn"):
                st.session_state["active_conv_index"] = None
                st.session_state["current_messages"]  = []
                st.session_state["chat_manager"]      = None
                st.rerun()

        convs = st.session_state["conversations"]
        if convs:
            for i, conv in enumerate(reversed(convs)):
                real_i    = len(convs) - 1 - i
                is_active = st.session_state["active_conv_index"] == real_i
                label     = conv.get("title", f"Chat {real_i+1}")[:30]
                turns     = conv.get("turn_count", 0)
                em_dist   = conv.get("emotion_dist", {})
                top_em    = max(em_dist, key=em_dist.get) if em_dist else ""
                em_icons  = {"neutral":"😐","joy":"😊","sadness":"😢","anger":"😠",
                             "fear":"😰","surprise":"😲","disgust":"😒"}
                em_icon   = em_icons.get(top_em, "💬")

                bg    = "#1e3a5f" if is_active else "#12161f"
                border= "#2563eb" if is_active else "#1e2435"
                color = "#93c5fd" if is_active else "#9ca3af"

                st.markdown(f"""
                <div style='background:{bg}; border:1px solid {border}; border-radius:8px;
                            padding:8px 12px; margin:3px 0;'>
                    <div style='font-size:12.5px; color:{color}; overflow:hidden;
                                text-overflow:ellipsis; white-space:nowrap;'>{em_icon} {label}</div>
                    <div style='font-size:10px; color:#4b5563; margin-top:2px;'>{turns} turns{f" · {top_em}" if top_em else ""}</div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("Open", key=f"hist_{real_i}", use_container_width=True):
                    st.session_state["active_conv_index"] = real_i
                    st.session_state["current_messages"]  = conv["messages"]
                    # Restore manager
                    from src.conversation.conversation_manager import ConversationManager
                    st.session_state["chat_manager"] = ConversationManager.from_dict(conv["manager_dict"])
                    st.rerun()
        else:
            st.markdown('<div style="font-size:11.5px; color:#374151; padding:4px;">No conversations yet.</div>',
                        unsafe_allow_html=True)

    st.divider()
    st.markdown('<div style="font-size:10px; color:#374151; padding:4px;">v1.0 · DistilBERT + Qwen2.5:3b</div>',
                unsafe_allow_html=True)

# ── Page routing ─────────────────────────────────────────────────────────────
if page_key == "chat":
    from src.pages.chat_page import render
    render()
elif page_key == "model_performance":
    from src.pages.model_performance_page import render
    render()
elif page_key == "about":
    from src.pages.about_page import render
    render()
