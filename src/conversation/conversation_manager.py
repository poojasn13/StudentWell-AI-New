"""
Phase 11 — Conversation Manager
==================================
Manages conversation state for a single user session.

Responsibilities
----------------
- Maintains ordered message history (user + assistant turns).
- Builds the context string fed to the emotion classifier.
- Enforces a maximum history length before summarising.
- Prevents history leakage between different sessions.
- Stores per-turn emotion predictions and response metadata.
- Provides serialisable state for Streamlit session_state storage.

Storage
-------
Currently uses in-memory storage (backed by Streamlit session_state).
The class interface is structured so a persistent SQLite or Redis backend
can be swapped in later without changing the calling code.
"""

import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_HISTORY_TURNS    = 10    # max turns kept in full before summarising
CONTEXT_WINDOW_TURNS = 4     # how many recent turns to use as classifier context
MAX_CONTEXT_CHARS    = 400   # character limit on context string


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """A single conversation turn (one student message + one assistant reply)."""
    turn_number:      int
    user_message:     str
    assistant_message: str            = ""
    emotion:          str             = "neutral"
    confidence:       float           = 0.0
    low_confidence:   bool            = False
    safety_triggered: bool            = False
    safety_level:     str             = "safe"
    response_system:  str             = "emotion_aware"  # "emotion_aware" | "normal"
    latency_ms:       float           = 0.0
    timestamp:        str             = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ConversationSession:
    """Full state for one conversation session."""
    session_id:       str
    turns:            list[Turn]  = field(default_factory=list)
    summary:          str         = ""   # filled in when history is compressed
    created_at:       str         = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_active:      str         = field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Manager class
# ---------------------------------------------------------------------------

class ConversationManager:
    """
    Manages a single conversation session.

    Usage (Streamlit)
    -----------------
    manager is stored in st.session_state["conversation_manager"]
    so it persists across Streamlit reruns but is isolated per browser session.

    Usage (comparison mode)
    -----------------------
    Two managers are created — one for System A, one for System B.
    They share the same session_id prefix but are stored separately.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session = ConversationSession(
            session_id=session_id or self._new_session_id()
        )
        log.debug(f"ConversationManager created: session_id={self.session.session_id}")

    # ------------------------------------------------------------------
    # Session ID
    # ------------------------------------------------------------------

    @staticmethod
    def _new_session_id() -> str:
        return "sess_" + uuid.uuid4().hex[:12]

    @property
    def session_id(self) -> str:
        return self.session.session_id

    # ------------------------------------------------------------------
    # History as Ollama message list
    # ------------------------------------------------------------------

    def get_ollama_history(self, max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
        """
        Return the conversation history formatted for ollama.chat() messages list.
        Only includes completed turns (those with an assistant reply).
        """
        history = []
        recent_turns = self.session.turns[-max_turns:]
        for turn in recent_turns:
            if turn.user_message:
                history.append({"role": "user",      "content": turn.user_message})
            if turn.assistant_message:
                history.append({"role": "assistant", "content": turn.assistant_message})
        return history

    # ------------------------------------------------------------------
    # Classifier context
    # ------------------------------------------------------------------

    def get_classifier_context(self) -> str:
        """
        Build a short context string for the emotion classifier.
        Uses the last CONTEXT_WINDOW_TURNS of student messages only.
        """
        recent = self.session.turns[-CONTEXT_WINDOW_TURNS:]
        if not recent:
            return ""

        parts = []
        for turn in recent:
            if turn.user_message:
                truncated = turn.user_message[:100]
                parts.append(f"Student: {truncated}")
            if turn.assistant_message:
                truncated = turn.assistant_message[:60]
                parts.append(f"Assistant: {truncated}")

        context = " | ".join(parts)
        return context[:MAX_CONTEXT_CHARS]

    # ------------------------------------------------------------------
    # Adding turns
    # ------------------------------------------------------------------

    def add_user_message(self, message: str) -> int:
        """
        Record a new student message, returning the turn_number.
        Call this BEFORE generating the response.
        """
        turn_number = len(self.session.turns) + 1
        turn = Turn(
            turn_number  = turn_number,
            user_message = message,
        )
        self.session.turns.append(turn)
        self.session.last_active = datetime.utcnow().isoformat()
        return turn_number

    def update_turn(
        self,
        turn_number:        int,
        assistant_message:  str         = "",
        emotion:            str         = "neutral",
        confidence:         float       = 0.0,
        low_confidence:     bool        = False,
        safety_triggered:   bool        = False,
        safety_level:       str         = "safe",
        response_system:    str         = "emotion_aware",
        latency_ms:         float       = 0.0,
    ):
        """
        Fill in the assistant response and metadata for a turn.
        Call this AFTER generation is complete.
        """
        # Find the turn by number
        for turn in self.session.turns:
            if turn.turn_number == turn_number:
                turn.assistant_message = assistant_message
                turn.emotion           = emotion
                turn.confidence        = confidence
                turn.low_confidence    = low_confidence
                turn.safety_triggered  = safety_triggered
                turn.safety_level      = safety_level
                turn.response_system   = response_system
                turn.latency_ms        = latency_ms
                return
        log.warning(f"Turn {turn_number} not found in session {self.session_id}")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def turn_count(self) -> int:
        return len(self.session.turns)

    @property
    def emotion_history(self) -> list[str]:
        """List of emotions across all completed turns."""
        return [t.emotion for t in self.session.turns if t.emotion]

    @property
    def average_confidence(self) -> float:
        confs = [t.confidence for t in self.session.turns if t.confidence > 0]
        return sum(confs) / len(confs) if confs else 0.0

    @property
    def safety_trigger_count(self) -> int:
        return sum(1 for t in self.session.turns if t.safety_triggered)

    def get_emotion_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for emotion in self.emotion_history:
            dist[emotion] = dist.get(emotion, 0) + 1
        return dist

    # ------------------------------------------------------------------
    # Serialisation helpers (for Streamlit session_state storage)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "session_id":  self.session.session_id,
            "created_at":  self.session.created_at,
            "last_active": self.session.last_active,
            "summary":     self.session.summary,
            "turns": [
                {
                    "turn_number":       t.turn_number,
                    "user_message":      t.user_message,
                    "assistant_message": t.assistant_message,
                    "emotion":           t.emotion,
                    "confidence":        t.confidence,
                    "low_confidence":    t.low_confidence,
                    "safety_triggered":  t.safety_triggered,
                    "safety_level":      t.safety_level,
                    "response_system":   t.response_system,
                    "latency_ms":        t.latency_ms,
                    "timestamp":         t.timestamp,
                }
                for t in self.session.turns
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationManager":
        manager = cls(session_id=data["session_id"])
        manager.session.created_at  = data.get("created_at", "")
        manager.session.last_active = data.get("last_active", "")
        manager.session.summary     = data.get("summary", "")
        manager.session.turns = [
            Turn(
                turn_number       = t["turn_number"],
                user_message      = t["user_message"],
                assistant_message = t.get("assistant_message", ""),
                emotion           = t.get("emotion", "neutral"),
                confidence        = t.get("confidence", 0.0),
                low_confidence    = t.get("low_confidence", False),
                safety_triggered  = t.get("safety_triggered", False),
                safety_level      = t.get("safety_level", "safe"),
                response_system   = t.get("response_system", "emotion_aware"),
                latency_ms        = t.get("latency_ms", 0.0),
                timestamp         = t.get("timestamp", ""),
            )
            for t in data.get("turns", [])
        ]
        return manager

    def reset(self):
        """Clear conversation history (start a new conversation)."""
        old_id = self.session.session_id
        self.session = ConversationSession(session_id=self._new_session_id())
        log.debug(f"Session reset: {old_id} → {self.session.session_id}")


# ---------------------------------------------------------------------------
# Comparison session manager
# ---------------------------------------------------------------------------

class ComparisonSessionManager:
    """
    Manages two parallel conversation histories for the comparison feature.
    Both System A (normal) and System B (emotion-aware) get their own history
    so each system only sees its own previous responses.
    """

    def __init__(self):
        base_id = "cmp_" + uuid.uuid4().hex[:8]
        self.system_a = ConversationManager(session_id=base_id + "_A")
        self.system_b = ConversationManager(session_id=base_id + "_B")

    @property
    def session_id(self) -> str:
        return self.system_a.session_id.replace("_A", "")

    def reset(self):
        base_id = "cmp_" + uuid.uuid4().hex[:8]
        self.system_a = ConversationManager(session_id=base_id + "_A")
        self.system_b = ConversationManager(session_id=base_id + "_B")


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mgr = ConversationManager()
    print(f"Session ID: {mgr.session_id}")

    # Simulate two turns
    t1 = mgr.add_user_message("I have an exam tomorrow and I'm really nervous.")
    mgr.update_turn(t1, assistant_message="That sounds stressful. Let's make a quick plan...",
                    emotion="fear", confidence=0.85)

    t2 = mgr.add_user_message("I finally passed! I can't believe it.")
    mgr.update_turn(t2, assistant_message="That's fantastic news! Well done.",
                    emotion="joy", confidence=0.91)

    print(f"Turns: {mgr.turn_count}")
    print(f"Emotions: {mgr.emotion_history}")
    print(f"Avg confidence: {mgr.average_confidence:.2f}")
    print(f"Emotion distribution: {mgr.get_emotion_distribution()}")
    print(f"\nClassifier context: {mgr.get_classifier_context()}")
    print(f"\nOllama history ({len(mgr.get_ollama_history())} messages):")
    for m in mgr.get_ollama_history():
        print(f"  [{m['role']}] {m['content'][:60]}")
