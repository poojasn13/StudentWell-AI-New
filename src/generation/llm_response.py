"""
Phase 7 — LLM Response Generation
====================================
Calls Qwen2.5:3b via Ollama to generate student support responses.

Provides two generation modes:
  - generate_emotion_aware_response()  — System B (emotion-guided)
  - generate_normal_response()         — System A (plain chatbot)

Both return a ResponseResult with the text, latency, and metadata.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import ollama

from src.generation.prompt_builder import (
    build_emotion_aware_messages,
    build_normal_messages,
    summarise_history_for_context,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_MODEL      = "qwen2.5:3b"
TEMPERATURE       = 0.7     # balanced creativity + consistency
TOP_P             = 0.9
MAX_TOKENS        = 512     # keeps responses focused
TIMEOUT_SECONDS   = 60      # Ollama call timeout guard


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ResponseResult:
    """
    Container for a generated response and its metadata.

    Fields
    ------
    text            : the generated response text
    system          : "emotion_aware" | "normal"
    emotion         : detected emotion (empty string for normal system)
    confidence      : classifier confidence (0.0 for normal system)
    low_confidence  : whether the classifier was uncertain
    latency_ms      : response generation time in milliseconds
    model           : Ollama model name used
    error           : error message if generation failed, else None
    prompt_tokens   : token count from Ollama (if available)
    """
    text:           str
    system:         str
    emotion:        str        = ""
    confidence:     float      = 0.0
    low_confidence: bool       = False
    latency_ms:     float      = 0.0
    model:          str        = OLLAMA_MODEL
    error:          Optional[str] = None
    prompt_tokens:  int        = 0

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.text.strip())


# ---------------------------------------------------------------------------
# Core call
# ---------------------------------------------------------------------------

def _call_ollama(messages: list[dict]) -> tuple[str, float, int]:
    """
    Call Ollama and return (response_text, latency_ms, prompt_tokens).
    Raises on failure — callers handle exceptions.
    """
    start = time.perf_counter()
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={
            "temperature": TEMPERATURE,
            "top_p":       TOP_P,
            "num_predict": MAX_TOKENS,
        },
    )
    latency_ms = (time.perf_counter() - start) * 1000

    text          = response["message"]["content"].strip()
    prompt_tokens = response.get("prompt_eval_count", 0) or 0
    return text, latency_ms, prompt_tokens


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_emotion_aware_response(
    user_message:         str,
    conversation_history: list[dict],
    emotion:              str,
    confidence:           float,
    low_confidence:       bool,
) -> ResponseResult:
    """
    Generate an emotion-aware response using Qwen2.5:3b (System B).

    Parameters
    ----------
    user_message          : current student message
    conversation_history  : list of previous {"role": ..., "content": ...}
    emotion               : predicted emotion label
    confidence            : classifier confidence score
    low_confidence        : True if confidence < threshold

    Returns
    -------
    ResponseResult
    """
    try:
        messages = build_emotion_aware_messages(
            user_message         = user_message,
            conversation_history = conversation_history,
            emotion              = emotion,
            confidence           = confidence,
            low_confidence       = low_confidence,
        )
        text, latency_ms, tokens = _call_ollama(messages)
        log.info(f"[emotion_aware] emotion={emotion} conf={confidence:.2f} latency={latency_ms:.0f}ms tokens={tokens}")
        return ResponseResult(
            text           = text,
            system         = "emotion_aware",
            emotion        = emotion,
            confidence     = confidence,
            low_confidence = low_confidence,
            latency_ms     = latency_ms,
            model          = OLLAMA_MODEL,
            prompt_tokens  = tokens,
        )
    except Exception as exc:
        log.error(f"[emotion_aware] generation failed: {exc}")
        return ResponseResult(
            text   = "",
            system = "emotion_aware",
            error  = str(exc),
        )


def generate_normal_response(
    user_message:         str,
    conversation_history: list[dict],
) -> ResponseResult:
    """
    Generate a standard (non-emotion-aware) response using Qwen2.5:3b (System A).

    Parameters
    ----------
    user_message          : current student message
    conversation_history  : list of previous {"role": ..., "content": ...}

    Returns
    -------
    ResponseResult
    """
    try:
        messages = build_normal_messages(
            user_message         = user_message,
            conversation_history = conversation_history,
        )
        text, latency_ms, tokens = _call_ollama(messages)
        log.info(f"[normal] latency={latency_ms:.0f}ms tokens={tokens}")
        return ResponseResult(
            text          = text,
            system        = "normal",
            latency_ms    = latency_ms,
            model         = OLLAMA_MODEL,
            prompt_tokens = tokens,
        )
    except Exception as exc:
        log.error(f"[normal] generation failed: {exc}")
        return ResponseResult(
            text   = "",
            system = "normal",
            error  = str(exc),
        )


def generate_both_responses(
    user_message:         str,
    conversation_history_a: list[dict],
    conversation_history_b: list[dict],
    emotion:              str,
    confidence:           float,
    low_confidence:       bool,
) -> tuple[ResponseResult, ResponseResult]:
    """
    Generate responses from both System A (normal) and System B (emotion-aware)
    for the side-by-side comparison feature.

    Returns (normal_result, emotion_aware_result).

    Both systems receive their own conversation histories so the comparison
    is independently maintained.
    """
    normal_result = generate_normal_response(user_message, conversation_history_a)
    aware_result  = generate_emotion_aware_response(
        user_message, conversation_history_b, emotion, confidence, low_confidence
    )
    return normal_result, aware_result


# ---------------------------------------------------------------------------
# Ollama health check
# ---------------------------------------------------------------------------

def check_ollama_available() -> tuple[bool, str]:
    """
    Return (is_available, message).
    Used by the Streamlit UI to show a friendly error when Ollama is down.
    """
    try:
        models = ollama.list()
        model_names = [m.get("name", m.get("model", "")) for m in models.get("models", [])]
        if OLLAMA_MODEL in model_names or any(OLLAMA_MODEL in n for n in model_names):
            return True, f"Ollama is running. Model '{OLLAMA_MODEL}' is available."
        else:
            return False, (
                f"Ollama is running but '{OLLAMA_MODEL}' is not found. "
                f"Available models: {model_names}. "
                f"Run: ollama pull {OLLAMA_MODEL}"
            )
    except Exception as exc:
        return False, (
            f"Cannot connect to Ollama: {exc}. "
            "Make sure Ollama is running: ollama serve"
        )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    available, msg = check_ollama_available()
    print(f"Ollama status: {msg}\n")
    if not available:
        exit(1)

    history = [
        {"role": "user",      "content": "I failed my mid-term and I feel terrible."},
        {"role": "assistant", "content": "I'm really sorry to hear that. Failing an exam is hard..."},
    ]
    message = "I don't even know if I should continue this course."

    print("--- Testing emotion-aware response (sadness) ---")
    result = generate_emotion_aware_response(
        user_message=message,
        conversation_history=history,
        emotion="sadness",
        confidence=0.78,
        low_confidence=False,
    )
    print(f"Success: {result.success}")
    print(f"Latency: {result.latency_ms:.0f}ms")
    print(f"Response:\n{result.text}\n")

    print("--- Testing normal response ---")
    result_n = generate_normal_response(message, history)
    print(f"Success: {result_n.success}")
    print(f"Latency: {result_n.latency_ms:.0f}ms")
    print(f"Response:\n{result_n.text}\n")
