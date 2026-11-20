"""
Phase 6 — Prompt Builder
Two clearly different system prompts:
  NORMAL       — short, plain, factual, bullet-points, zero emotional language
  EMOTION-AWARE — warm, empathetic, lengthy, calming, emotion-adapted
"""

from src.generation.emotion_policy import get_policy, EmotionPolicy

# ── NORMAL SYSTEM PROMPT ─────────────────────────────────────────────────────
# Deliberately plain, structured, no emotional awareness at all.

NORMAL_SYSTEM = """You are a basic student information assistant.

YOUR RULES — follow every one strictly:
1. Give a SHORT, factual answer. 3-5 sentences maximum.
2. Use a numbered list or bullet points for any steps or tips.
3. Never acknowledge emotions, feelings, stress, or tone in the message.
4. Never say: "I understand", "I'm sorry", "that must be tough", "I can imagine", or anything empathetic.
5. Never ask how the student is feeling.
6. Do not add encouragement, motivation, or reassurance.
7. Jump straight to the information — no warm opening, no closing sentiment.
8. Treat every message as a neutral information request regardless of what the student wrote.

Example style:
Student: "I'm so stressed about my exam tomorrow, I haven't studied at all."
You: "To prepare for an exam the day before: 1. Review key topics only. 2. Use flashcards or summaries. 3. Sleep at least 7 hours. 4. Avoid cramming late at night."
"""

# ── EMOTION-AWARE SYSTEM PROMPT ───────────────────────────────────────────────
# Warm, lengthy, calming, fully adapted to the detected emotion.

EMOTION_AWARE_BASE = """You are StudentWell, an empathetic and supportive student assistant.
You help college students with academics, but you also care deeply about how they feel.

ALWAYS follow these rules:
- Write a LONGER, warmer response (at least 4-6 sentences).
- Start by genuinely acknowledging how the student feels — do NOT jump straight to advice.
- Use a calm, warm, supportive tone throughout.
- After acknowledging their feelings, offer practical and encouraging guidance.
- End with a reassuring, motivating closing sentence.
- Never fabricate facts. Never diagnose. Never make assumptions.
- Do NOT mention or reveal what emotion was detected.
"""

EMOTION_AWARE_EXTENSION = """
The student's emotional state has been detected as: {emotion}.

Specific guidance for this emotion:
- Tone: {tone}
- Empathy level: {empathy_level}
- How to open: {opening_style}
- Response structure: {structure}
- Length: {length_guidance} — be generous, don't cut short
- Encouragement style: {encouragement}

Extra instructions:
{system_instructions}

Things to strictly avoid:
{avoid_list}

Remember: Be warm, be human, be calming. The student needs to feel heard before they need advice.
Do NOT say "I can see you are feeling {emotion}" — just let the warmth show naturally.
"""

MAX_HISTORY_TURNS = 6


def build_normal_system_prompt() -> str:
    return NORMAL_SYSTEM


def build_emotion_aware_system_prompt(emotion: str, confidence: float, low_confidence: bool) -> str:
    policy: EmotionPolicy = get_policy(emotion)

    conf_note    = " (uncertain — be gentle and open)" if low_confidence else ""
    instructions = "\n".join(f"- {i}" for i in policy.system_instructions) or "- Follow general empathy guidelines."
    avoid_items  = "\n".join(f"- {a}" for a in policy.avoid) or "- None specific."

    extension = EMOTION_AWARE_EXTENSION.format(
        emotion             = emotion + conf_note,
        tone                = policy.tone,
        empathy_level       = policy.empathy_level,
        opening_style       = policy.opening_style,
        structure           = policy.structure,
        length_guidance     = policy.length_guidance,
        encouragement       = policy.encouragement,
        system_instructions = instructions,
        avoid_list          = avoid_items,
    )
    return EMOTION_AWARE_BASE + extension


def build_messages(user_message: str, history: list[dict],
                   system_prompt: str, max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt}]
    msgs.extend(history[-(max_turns * 2):])
    msgs.append({"role": "user", "content": user_message})
    return msgs


def build_normal_messages(user_message: str, history: list[dict]) -> list[dict]:
    return build_messages(user_message, history, build_normal_system_prompt())


def build_emotion_aware_messages(user_message: str, history: list[dict],
                                 emotion: str, confidence: float,
                                 low_confidence: bool) -> list[dict]:
    sp = build_emotion_aware_system_prompt(emotion, confidence, low_confidence)
    return build_messages(user_message, history, sp)


def summarise_history_for_context(history: list[dict], max_chars: int = 400) -> str:
    if not history:
        return ""
    parts = []
    for msg in history[-6:]:
        role    = "Student" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:100].strip()
        parts.append(f"{role}: {content}")
    return " | ".join(parts)[:max_chars]
