"""
Phase 10 — Safety Checker
===========================
A conservative, rule-based safety layer that runs BEFORE normal LLM
generation and BEFORE emotion classification.

Design principles
-----------------
1.  Keyword + pattern matching — fast, transparent, testable.
2.  Conservative: false positives (over-triggering) are preferable to
    false negatives (missing a genuine crisis signal).
3.  Three severity tiers:
      CRISIS      — potential immediate self-harm risk → crisis response
      CONCERN     — distress signals that warrant gentle check-in
      SAFE        — no safety signal detected → normal processing
4.  Does NOT diagnose.
5.  Does NOT claim to provide counselling.
6.  Does NOT use DistilBERT emotion labels to detect self-harm risk.
7.  Keyword-based detection has known limitations — documented below.
8.  Does not store raw messages unnecessarily.

Limitations (must be disclosed)
--------------------------------
- Keyword matching cannot understand context, sarcasm, or metaphor.
  "I could kill for a coffee" is flagged but is not a crisis signal.
- It will miss signals expressed in indirect, coded, or non-English language.
- It is a safety NET, not a replacement for human professional support.
- Students in genuine crisis must be directed to qualified professionals.
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------

class SafetyLevel(str, Enum):
    SAFE    = "safe"
    CONCERN = "concern"
    CRISIS  = "crisis"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SafetyResult:
    level:           SafetyLevel
    triggered:       bool
    matched_patterns: list[str]  = field(default_factory=list)
    response_text:   str         = ""   # pre-written response (non-empty when triggered)
    log_code:        str         = ""   # short code for analytics logging


# ---------------------------------------------------------------------------
# Pattern sets
# ---------------------------------------------------------------------------

# CRISIS patterns — explicit self-harm, suicidal ideation, immediate danger
CRISIS_PATTERNS = [
    r"\bkill\s+myself\b",
    r"\bwant\s+to\s+die\b",
    r"\bending\s+my\s+life\b",
    r"\bend\s+it\s+all\b",
    r"\bsuicid\w*\b",
    r"\bself[\s\-]harm\b",
    r"\bcut\s+myself\b",
    r"\bhurt\s+myself\b",
    r"\bno\s+reason\s+to\s+(live|go\s+on)\b",
    r"\bdon'?t\s+want\s+to\s+(be\s+here|exist)\s+anymore\b",
    r"\bthink\s+about\s+ending\b",
    r"\bthought\s+about\s+(killing|hurting)\s+myself\b",
    r"\bnobody\s+would\s+miss\s+me\b",
    r"\bwould\s+be\s+better\s+off\s+(without\s+me|dead)\b",
    r"\boverdose\b",
    r"\bwant\s+to\s+disappear\s+forever\b",
]

# CONCERN patterns — significant distress signals warranting a gentle check-in
# (not necessarily crisis, but worth acknowledging directly)
CONCERN_PATTERNS = [
    r"\bcompletely\s+(hopeless|worthless|useless)\b",
    r"\bcan'?t\s+(go\s+on|take\s+it\s+anymore|cope)\b",
    r"\bgive\s+up\s+on\s+(everything|life|all)\b",
    r"\bno\s+point\s+in\s+(anything|trying)\b",
    r"\bfeel\s+so\s+alone\b",
    r"\bcompletely\s+broken\b",
    r"\bhate\s+myself\b",
    r"\bdeserve\s+to\s+(suffer|be\s+punished)\b",
    r"\bi\s+am\s+nothing\b",
    r"\bwish\s+i\s+(was|were)\s+(never\s+born|dead)\b",
    r"\bexhausted\s+(of\s+life|of\s+everything)\b",
]

# ---------------------------------------------------------------------------
# Pre-written safe responses
# ---------------------------------------------------------------------------

CRISIS_RESPONSE = """I'm really glad you reached out, and I want you to know what you're feeling matters.

What you've shared sounds very serious, and it's beyond what I'm equipped to help with. Please reach out to someone who can truly support you right now:

**If you are in immediate danger:** Contact your local emergency services (such as 112 or 911) or go to the nearest emergency department.

**Crisis support lines (available 24/7):**
- iCall (India): 9152987821
- Vandrevala Foundation: 1860-2662-345
- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

**Within your college:** Please consider speaking to a counsellor, a trusted faculty member, or a friend today — not because you have to have all the answers, but because you don't have to carry this alone.

I'm an academic assistant and I'm not able to provide the support you deserve right now. Please talk to someone. You matter."""

CONCERN_RESPONSE = """I want to pause for a moment, because what you've shared sounds like you're carrying something really heavy right now.

Before we talk about anything academic — how are you doing? It's completely okay if things feel overwhelming.

If you're going through a difficult time beyond just academic stress, please consider talking to someone you trust — a friend, family member, or your college's student support or counselling service. You don't have to manage everything alone.

I'm here to help with academic questions whenever you're ready, but your wellbeing comes first."""

# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

class SafetyChecker:
    """
    Rule-based safety checker.  Instantiate once and call check() per message.
    """

    def __init__(self):
        self._crisis_re  = [re.compile(p, re.IGNORECASE) for p in CRISIS_PATTERNS]
        self._concern_re = [re.compile(p, re.IGNORECASE) for p in CONCERN_PATTERNS]

    def check(self, text: str) -> SafetyResult:
        """
        Check a student message for safety signals.

        Parameters
        ----------
        text : the student's message text

        Returns
        -------
        SafetyResult with level, triggered flag, and pre-written response
        if applicable.
        """
        if not text or not text.strip():
            return SafetyResult(level=SafetyLevel.SAFE, triggered=False)

        # Check CRISIS first (higher priority)
        crisis_matches = [
            p.pattern for p in self._crisis_re if p.search(text)
        ]
        if crisis_matches:
            log.warning(f"SAFETY CRISIS triggered — patterns: {crisis_matches[:3]}")
            return SafetyResult(
                level             = SafetyLevel.CRISIS,
                triggered         = True,
                matched_patterns  = crisis_matches,
                response_text     = CRISIS_RESPONSE,
                log_code          = "CRISIS",
            )

        # Check CONCERN
        concern_matches = [
            p.pattern for p in self._concern_re if p.search(text)
        ]
        if concern_matches:
            log.info(f"SAFETY CONCERN triggered — patterns: {concern_matches[:3]}")
            return SafetyResult(
                level             = SafetyLevel.CONCERN,
                triggered         = True,
                matched_patterns  = concern_matches,
                response_text     = CONCERN_RESPONSE,
                log_code          = "CONCERN",
            )

        return SafetyResult(level=SafetyLevel.SAFE, triggered=False, log_code="SAFE")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_checker_instance: SafetyChecker = None


def get_safety_checker() -> SafetyChecker:
    global _checker_instance
    if _checker_instance is None:
        _checker_instance = SafetyChecker()
    return _checker_instance


def check_safety(text: str) -> SafetyResult:
    """Module-level convenience function."""
    return get_safety_checker().check(text)


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

TEST_CASES = [
    # (text, expected_level)
    ("I want to kill myself",                               SafetyLevel.CRISIS),
    ("I've been having thoughts about ending my life",      SafetyLevel.CRISIS),
    ("I feel like nobody would miss me",                    SafetyLevel.CRISIS),
    ("I hate myself and I completely deserve to suffer",    SafetyLevel.CONCERN),
    ("I can't take it anymore, I feel so alone",            SafetyLevel.CONCERN),
    ("I failed my exam and I'm really stressed",            SafetyLevel.SAFE),
    ("I am so sad about my grades",                         SafetyLevel.SAFE),
    ("I could kill for a coffee right now",                 SafetyLevel.SAFE),   # false positive test
    ("What resources does the library have?",               SafetyLevel.SAFE),
    ("I'm angry at my professor",                           SafetyLevel.SAFE),
    ("I want to die laughing at this meme",                 SafetyLevel.CRISIS), # conservative — over-trigger
]


def run_tests():
    checker = SafetyChecker()
    passed = 0
    failed = 0

    print("=" * 55)
    print("SAFETY CHECKER TEST SUITE")
    print("=" * 55)
    for text, expected in TEST_CASES:
        result = checker.check(text)
        ok     = result.level == expected
        mark   = "✓" if ok else "✗"
        status = "PASS" if ok else f"FAIL (got {result.level})"
        print(f"  {mark} [{expected.value:<8}] {status}")
        print(f"      Text: {text[:60]}")
        if passed_test := ok:
            passed += 1
        else:
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {len(TEST_CASES)}")
    print("\nNote: 'I want to die laughing' is intentionally over-triggered.")
    print("Keyword-based detection cannot understand context — this is a known limitation.")
    print("=" * 55)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_tests()
