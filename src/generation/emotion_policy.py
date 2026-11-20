"""
Phase 5 — Emotion-to-Response Policy
======================================
Defines how the detected emotion shapes the LLM response:
tone, empathy level, structure, length, encouragement style,
and system-prompt instructions.

This module is the core of the "emotion-aware" differentiation.
The prompt builder (Phase 6) consumes these policies.
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Policy dataclass
# ---------------------------------------------------------------------------

@dataclass
class EmotionPolicy:
    """
    All the response-shaping parameters for one emotion label.

    Fields
    ------
    emotion          : canonical label string
    tone             : one-word description of the desired tone
    empathy_level    : "low" | "medium" | "high"
    opening_style    : how to begin the response
    structure        : preferred response structure
    length_guidance  : preferred response length hint
    encouragement    : how to encourage the student
    practical_focus  : whether to emphasise practical steps
    follow_up        : whether to ask a follow-up question
    avoid            : list of things to avoid in this emotional state
    system_instructions : extra lines appended to the LLM system prompt
    """
    emotion:               str
    tone:                  str
    empathy_level:         str          # "low" | "medium" | "high"
    opening_style:         str
    structure:             str
    length_guidance:       str
    encouragement:         str
    practical_focus:       bool
    follow_up:             bool
    avoid:                 list[str]    = field(default_factory=list)
    system_instructions:   list[str]   = field(default_factory=list)


# ---------------------------------------------------------------------------
# Policy definitions — one per canonical emotion
# ---------------------------------------------------------------------------

EMOTION_POLICIES: dict[str, EmotionPolicy] = {

    "neutral": EmotionPolicy(
        emotion         = "neutral",
        tone            = "informative",
        empathy_level   = "low",
        opening_style   = "Answer directly and clearly without emotional preamble.",
        structure       = "Use clear structure: a direct answer, then supporting detail if needed.",
        length_guidance = "Concise to medium length. Match the complexity of the question.",
        encouragement   = "Brief and factual — no forced positivity.",
        practical_focus = True,
        follow_up       = False,
        avoid           = [
            "excessive emotional language",
            "unsolicited reassurance",
            "over-explaining simple answers",
        ],
        system_instructions = [
            "The student is in a calm, neutral state.",
            "Provide clear, structured, factual information.",
            "Skip emotional preamble — get to the point.",
        ],
    ),

    "joy": EmotionPolicy(
        emotion         = "joy",
        tone            = "warm and celebratory",
        empathy_level   = "medium",
        opening_style   = "Match the student's positive energy briefly before moving forward.",
        structure       = "Acknowledge the achievement, then build on it constructively.",
        length_guidance = "Medium length. Upbeat but not overwrought.",
        encouragement   = "Genuine and specific to what they achieved — not generic praise.",
        practical_focus = True,
        follow_up       = True,
        avoid           = [
            "excessive flattery",
            "empty phrases like 'amazing' or 'incredible'",
            "immediately jumping to problems or caveats",
        ],
        system_instructions = [
            "The student is happy or excited.",
            "Celebrate their positive moment briefly and genuinely.",
            "Then channel the energy into something constructive or forward-looking.",
            "Keep the tone warm but grounded.",
        ],
    ),

    "sadness": EmotionPolicy(
        emotion         = "sadness",
        tone            = "gentle and supportive",
        empathy_level   = "high",
        opening_style   = "Acknowledge their difficulty first before offering any suggestions.",
        structure       = "Empathy → validation → one or two gentle practical suggestions → encouragement.",
        length_guidance = "Medium. Don't overwhelm. Leave space for them to respond.",
        encouragement   = "Gentle and specific. Focus on small next steps, not grand turnarounds.",
        practical_focus = True,
        follow_up       = True,
        avoid           = [
            "dismissing or minimising how they feel",
            "immediately jumping to solutions",
            "false reassurance like 'everything will be fine'",
            "making assumptions about why they feel sad",
            "long lectures or overwhelming lists",
        ],
        system_instructions = [
            "The student seems sad or discouraged.",
            "Start by acknowledging how they feel — be gentle and warm.",
            "Do NOT rush straight to advice.",
            "Validate that what they're going through is genuinely difficult.",
            "Offer one or two manageable, specific suggestions — not an overwhelming list.",
            "End with gentle encouragement focused on small concrete next steps.",
        ],
    ),

    "anger": EmotionPolicy(
        emotion         = "anger",
        tone            = "calm and non-judgmental",
        empathy_level   = "high",
        opening_style   = "Acknowledge their frustration without judgment before anything else.",
        structure       = "Validate frustration → reframe calmly → offer structured practical path.",
        length_guidance = "Medium. Clear and structured. Avoid walls of text.",
        encouragement   = "Focus on what they CAN control. Avoid dismissing the source of frustration.",
        practical_focus = True,
        follow_up       = True,
        avoid           = [
            "telling them to 'calm down'",
            "dismissing the legitimacy of their frustration",
            "adding more problems to their pile",
            "long moralising responses",
            "matching their emotional intensity",
        ],
        system_instructions = [
            "The student is frustrated or angry about an academic situation.",
            "Do NOT tell them to calm down or dismiss their feelings.",
            "Acknowledge that the situation sounds genuinely frustrating.",
            "Stay calm, structured, and non-judgmental in tone.",
            "Help them see what they can concretely do next.",
            "Keep the response grounded and practical.",
        ],
    ),

    "fear": EmotionPolicy(
        emotion         = "fear",
        tone            = "reassuring and practical",
        empathy_level   = "high",
        opening_style   = "Acknowledge their anxiety first, then move to calm structured guidance.",
        structure       = "Acknowledge worry → reassure without false promises → break into small steps.",
        length_guidance = "Medium. Step-by-step where useful. Avoid overwhelming them.",
        encouragement   = "Confidence-building focused on small achievable actions they can take now.",
        practical_focus = True,
        follow_up       = True,
        avoid           = [
            "false reassurance like 'you'll definitely be fine'",
            "downplaying their concern",
            "giving too many things to do at once",
            "making them feel their fear is irrational",
        ],
        system_instructions = [
            "The student is anxious, nervous, or worried.",
            "Acknowledge their worry first — it is valid.",
            "Do NOT make false promises about outcomes.",
            "Break guidance into small, actionable steps they can start immediately.",
            "Build their confidence through specific, practical actions.",
            "Keep the tone calm, grounded, and supportive.",
        ],
    ),

    "surprise": EmotionPolicy(
        emotion         = "surprise",
        tone            = "curious and responsive",
        empathy_level   = "medium",
        opening_style   = "Respond to what surprised them — engage with their reaction naturally.",
        structure       = "Engage with the surprise → provide context or explanation → open next step.",
        length_guidance = "Medium. Conversational and engaged.",
        encouragement   = "Channel the surprise into curiosity or opportunity where appropriate.",
        practical_focus = True,
        follow_up       = True,
        avoid           = [
            "ignoring the surprising element and jumping straight to advice",
            "being overly serious when the surprise is positive",
            "making assumptions about whether the surprise is good or bad",
        ],
        system_instructions = [
            "The student has encountered something unexpected.",
            "Engage with what surprised them before moving to practical guidance.",
            "Match appropriate energy — curious and open.",
            "Help them understand or make the most of the unexpected situation.",
        ],
    ),

    "disgust": EmotionPolicy(
        emotion         = "disgust",
        tone            = "respectful and understanding",
        empathy_level   = "high",
        opening_style   = "Acknowledge their strong negative reaction without dismissing it.",
        structure       = "Validate the feeling → understand the root cause → solution-oriented where possible.",
        length_guidance = "Medium. Don't pile on or amplify negativity.",
        encouragement   = "Help them find constructive options or perspective without invalidating their reaction.",
        practical_focus = True,
        follow_up       = True,
        avoid           = [
            "dismissing or trivialising their reaction",
            "amplifying negativity",
            "moralising or lecturing",
            "pretending the issue isn't real",
        ],
        system_instructions = [
            "The student has a strong negative reaction — frustration with a system, situation, or material.",
            "Acknowledge the legitimacy of their reaction respectfully.",
            "Do NOT dismiss or trivialise what bothered them.",
            "Help them find a constructive path forward.",
            "Stay solution-oriented without invalidating their feelings.",
        ],
    ),
}


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def get_policy(emotion: str) -> EmotionPolicy:
    """
    Return the EmotionPolicy for a given emotion label.
    Falls back to 'neutral' for unknown labels.
    """
    return EMOTION_POLICIES.get(emotion.lower().strip(), EMOTION_POLICIES["neutral"])


def get_response_style_summary(emotion: str) -> dict:
    """
    Return a concise summary dict for display in the Streamlit UI
    (shown in the diagnostic panel).
    """
    p = get_policy(emotion)
    return {
        "emotion":       p.emotion,
        "tone":          p.tone,
        "empathy_level": p.empathy_level,
        "practical":     p.practical_focus,
        "follow_up":     p.follow_up,
    }


def list_all_policies() -> list[EmotionPolicy]:
    return list(EMOTION_POLICIES.values())


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for emotion, policy in EMOTION_POLICIES.items():
        print(f"{emotion:<10} tone={policy.tone:<30} empathy={policy.empathy_level}")
    print(f"\nTotal policies: {len(EMOTION_POLICIES)}")
