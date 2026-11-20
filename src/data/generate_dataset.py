"""
Phase 1 — Student Conversation Dataset Generator
=================================================
Generates a multi-turn student-focused conversational dataset using
Qwen2.5:3b via Ollama.  The dataset feeds DistilBERT emotion classification.

Target: ~8,990 samples across 7 emotions.
Each sample is one conversation turn (row).  Turns belong to multi-turn
conversations identified by conversation_id.

Fields
------
conversation_id : str   e.g. C0001
turn_id         : int   1-based index within the conversation
context         : str   summary of previous turns (empty for turn 1)
text            : str   the student message for this turn
emotion         : str   one of the 7 canonical labels

Run
---
    python src/data/generate_dataset.py

Checkpointing: progress is saved every CHECKPOINT_EVERY conversations so
the script can be safely interrupted and resumed.
"""

import json
import os
import random
import re
import time
import logging
from pathlib import Path
from typing import Optional

import ollama
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_MODEL = "qwen2.5:3b"
OUTPUT_DIR   = Path("data/raw")
OUTPUT_FILE  = OUTPUT_DIR / "student_conversations.csv"
CHECKPOINT_FILE = OUTPUT_DIR / "generation_checkpoint.json"
LOG_FILE     = Path("logs/dataset_generation.log")

# Target samples per emotion — ~4,000 total (~572 per emotion)
EMOTION_TARGETS = {
    "neutral":  572,
    "joy":      572,
    "sadness":  572,
    "anger":    572,
    "fear":     572,
    "surprise": 572,
    "disgust":  572,
}
TOTAL_TARGET = sum(EMOTION_TARGETS.values())   # 4,004

# Average turns per conversation — we target 3 turns on average,
# so we need roughly TOTAL_TARGET / 3 ≈ 2,997 conversations.
AVG_TURNS = 3

# How many conversations to generate per Ollama call
CONVERSATIONS_PER_BATCH = 1

# Save checkpoint every N conversations
CHECKPOINT_EVERY = 50

# Temperature for generation — higher = more diverse
GENERATION_TEMPERATURE = 0.85

# Max retries on parse failure
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scenario templates
# Each entry: (scenario_description, primary_emotion, secondary_emotions)
# secondary_emotions defines what emotion the LATER turns may shift to,
# creating realistic emotional arcs within a conversation.
# ---------------------------------------------------------------------------

SCENARIOS = [
    # --- Academic pressure ---
    {
        "category": "exam_pressure",
        "description": "Student preparing for a difficult upcoming exam",
        "arc": [("fear", "fear"), ("fear", "sadness"), ("fear", "joy")],
    },
    {
        "category": "exam_pressure",
        "description": "Student feeling overwhelmed by multiple exams in one week",
        "arc": [("anger", "anger"), ("anger", "sadness"), ("sadness", "neutral")],
    },
    {
        "category": "exam_pressure",
        "description": "Student unexpectedly scored well in an exam they thought they failed",
        "arc": [("fear", "surprise"), ("surprise", "joy"), ("joy", "joy")],
    },
    # --- Assignments and deadlines ---
    {
        "category": "assignments",
        "description": "Student struggling to finish an assignment the night before the deadline",
        "arc": [("fear", "fear"), ("anger", "anger"), ("neutral", "neutral")],
    },
    {
        "category": "assignments",
        "description": "Student frustrated that their assignment requirements keep changing",
        "arc": [("anger", "anger"), ("disgust", "anger"), ("neutral", "neutral")],
    },
    {
        "category": "assignments",
        "description": "Student proud of completing a difficult assignment ahead of schedule",
        "arc": [("joy", "joy"), ("joy", "neutral"), ("neutral", "neutral")],
    },
    # --- Subject difficulty ---
    {
        "category": "subject_difficulty",
        "description": "Student unable to understand a core concept despite repeated attempts",
        "arc": [("frustration", "anger"), ("sadness", "sadness"), ("neutral", "joy")],
    },
    {
        "category": "subject_difficulty",
        "description": "Student confused about a topic after a confusing lecture",
        "arc": [("surprise", "neutral"), ("neutral", "neutral"), ("joy", "joy")],
    },
    {
        "category": "subject_difficulty",
        "description": "Student disgusted by an outdated textbook with wrong information",
        "arc": [("disgust", "anger"), ("anger", "neutral"), ("neutral", "neutral")],
    },
    # --- Placement and interviews ---
    {
        "category": "placement",
        "description": "Student anxious about an upcoming campus placement interview",
        "arc": [("fear", "fear"), ("fear", "neutral"), ("joy", "joy")],
    },
    {
        "category": "placement",
        "description": "Student rejected after multiple placement interviews",
        "arc": [("sadness", "sadness"), ("anger", "anger"), ("neutral", "neutral")],
    },
    {
        "category": "placement",
        "description": "Student surprised by receiving an unexpected job offer",
        "arc": [("surprise", "joy"), ("joy", "joy"), ("neutral", "neutral")],
    },
    # --- Internship and career ---
    {
        "category": "internship",
        "description": "Student uncertain about which career path to choose",
        "arc": [("neutral", "neutral"), ("fear", "fear"), ("neutral", "neutral")],
    },
    {
        "category": "internship",
        "description": "Student excited about starting their first internship",
        "arc": [("joy", "joy"), ("fear", "neutral"), ("joy", "joy")],
    },
    {
        "category": "internship",
        "description": "Student disappointed by a poor internship experience",
        "arc": [("sadness", "disgust"), ("disgust", "anger"), ("neutral", "neutral")],
    },
    # --- Project difficulties ---
    {
        "category": "projects",
        "description": "Student struggling with a group project where members are not contributing",
        "arc": [("anger", "anger"), ("disgust", "anger"), ("sadness", "neutral")],
    },
    {
        "category": "projects",
        "description": "Student excited about a creative project idea they just had",
        "arc": [("joy", "joy"), ("neutral", "neutral"), ("joy", "surprise")],
    },
    {
        "category": "projects",
        "description": "Student panicking because their project demo crashed",
        "arc": [("fear", "anger"), ("anger", "sadness"), ("neutral", "neutral")],
    },
    # --- Academic achievements ---
    {
        "category": "achievement",
        "description": "Student celebrating getting the highest score in the class",
        "arc": [("joy", "joy"), ("joy", "surprise"), ("joy", "neutral")],
    },
    {
        "category": "achievement",
        "description": "Student surprised by winning a college competition they did not expect to win",
        "arc": [("surprise", "joy"), ("joy", "joy"), ("neutral", "neutral")],
    },
    {
        "category": "achievement",
        "description": "Student proud but also tired after completing a semester project",
        "arc": [("joy", "neutral"), ("neutral", "sadness"), ("neutral", "neutral")],
    },
    # --- Friendships and group work ---
    {
        "category": "social",
        "description": "Student frustrated with a friend who keeps cancelling study plans",
        "arc": [("anger", "anger"), ("sadness", "sadness"), ("neutral", "neutral")],
    },
    {
        "category": "social",
        "description": "Student feeling left out of a friend group at college",
        "arc": [("sadness", "sadness"), ("anger", "sadness"), ("neutral", "neutral")],
    },
    {
        "category": "social",
        "description": "Student happy about making a new friend who shares their interests",
        "arc": [("joy", "joy"), ("joy", "neutral"), ("neutral", "neutral")],
    },
    # --- College activities ---
    {
        "category": "college_activities",
        "description": "Student nervous about performing in a college cultural event",
        "arc": [("fear", "fear"), ("neutral", "neutral"), ("joy", "joy")],
    },
    {
        "category": "college_activities",
        "description": "Student disappointed that their club proposal was rejected",
        "arc": [("sadness", "anger"), ("anger", "neutral"), ("neutral", "neutral")],
    },
    {
        "category": "college_activities",
        "description": "Student excited about an upcoming college trip or fest",
        "arc": [("joy", "joy"), ("joy", "neutral"), ("surprise", "joy")],
    },
    # --- Learning new skills ---
    {
        "category": "learning",
        "description": "Student struggling to learn a new programming language",
        "arc": [("frustration", "anger"), ("anger", "neutral"), ("joy", "joy")],
    },
    {
        "category": "learning",
        "description": "Student delighted that a concept they had been struggling with finally clicked",
        "arc": [("surprise", "joy"), ("joy", "joy"), ("neutral", "neutral")],
    },
    {
        "category": "learning",
        "description": "Student feeling inadequate compared to classmates who learn faster",
        "arc": [("sadness", "sadness"), ("anger", "neutral"), ("neutral", "neutral")],
    },
    # --- Failure and recovery ---
    {
        "category": "failure_recovery",
        "description": "Student failed an exam and is considering dropping the subject",
        "arc": [("sadness", "sadness"), ("anger", "sadness"), ("neutral", "neutral")],
    },
    {
        "category": "failure_recovery",
        "description": "Student disappointed after putting a lot of effort into something that did not work",
        "arc": [("sadness", "anger"), ("anger", "neutral"), ("neutral", "joy")],
    },
    {
        "category": "failure_recovery",
        "description": "Student slowly recovering confidence after a rough semester",
        "arc": [("sadness", "neutral"), ("neutral", "neutral"), ("joy", "joy")],
    },
    # --- Positive feedback ---
    {
        "category": "positive_feedback",
        "description": "Student received unexpected praise from a professor",
        "arc": [("surprise", "joy"), ("joy", "joy"), ("neutral", "neutral")],
    },
    {
        "category": "positive_feedback",
        "description": "Student encouraged after getting helpful feedback on their essay",
        "arc": [("joy", "neutral"), ("neutral", "joy"), ("joy", "joy")],
    },
    {
        "category": "positive_feedback",
        "description": "Student surprised to find their research cited by another student",
        "arc": [("surprise", "surprise"), ("joy", "joy"), ("neutral", "neutral")],
    },
    # --- General student conversations ---
    {
        "category": "general",
        "description": "Student asking general questions about managing college life",
        "arc": [("neutral", "neutral"), ("neutral", "neutral"), ("neutral", "neutral")],
    },
    {
        "category": "general",
        "description": "Student venting casually about a tiring college day",
        "arc": [("disgust", "neutral"), ("neutral", "neutral"), ("neutral", "neutral")],
    },
    {
        "category": "general",
        "description": "Student curious about resources available at their college",
        "arc": [("neutral", "surprise"), ("surprise", "neutral"), ("neutral", "neutral")],
    },
]

# ---------------------------------------------------------------------------
# Canonical labels — frustration maps to anger during generation but is
# listed separately in arc definitions for prompt realism
# ---------------------------------------------------------------------------

CANONICAL_EMOTIONS = {
    "neutral", "joy", "sadness", "anger", "fear", "surprise", "disgust"
}

EMOTION_ALIASES = {
    "frustration": "anger",
    "frustrated":  "anger",
    "happy":       "joy",
    "happiness":   "joy",
    "scared":      "fear",
    "afraid":      "fear",
    "anxious":     "fear",
    "anxiety":     "fear",
    "disgusted":   "disgust",
    "surprised":   "surprise",
    "worried":     "fear",
}


def normalize_emotion(label: str) -> Optional[str]:
    """Return canonical emotion label or None if unrecognisable."""
    label = label.strip().lower()
    if label in CANONICAL_EMOTIONS:
        return label
    return EMOTION_ALIASES.get(label)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a dataset creation assistant generating realistic multi-turn student conversations for an AI training dataset.

RULES:
1. Generate ONLY the JSON requested — no extra commentary.
2. Each conversation must have exactly {num_turns} turns.
3. Each turn is one student message — not a chatbot reply.
4. Each student message must sound natural, like a real college student typed it.
5. The emotion label for each turn must accurately reflect the content and tone.
6. Use varied sentence lengths, informal language where appropriate, contractions, and realistic phrasing.
7. Do NOT include the emotion word itself inside the student message text.
8. The context field for turn 1 must be an empty string "".
9. For later turns, write a short 1-sentence summary of the conversation so far as the context.
10. Each turn's text should be a complete thought — not a continuation mid-sentence.
11. Avoid repetitive phrasing across turns.
12. Valid emotion labels: neutral, joy, sadness, anger, fear, surprise, disgust.
"""

GENERATION_PROMPT = """Generate a realistic {num_turns}-turn student conversation for this scenario:

Scenario: {description}

Emotional arc (the student emotion for each turn):
{arc_str}

Output ONLY valid JSON in this exact format:
{{
  "turns": [
    {{
      "turn_id": 1,
      "context": "",
      "text": "<student message for turn 1>",
      "emotion": "<emotion label for turn 1>"
    }},
    {{
      "turn_id": 2,
      "context": "<one-sentence summary of turn 1>",
      "text": "<student message for turn 2>",
      "emotion": "<emotion label for turn 2>"
    }}
  ]
}}

Requirements:
- Student messages must sound genuine, like typed by a real college student.
- Do NOT include the emotion word in the message text.
- Use natural, slightly informal language.
- Each message should be 1–3 sentences.
- Some messages can be short (under 10 words), some longer.
"""


def build_prompt(scenario: dict, num_turns: int) -> str:
    arc = scenario["arc"][:num_turns]
    arc_str = "\n".join(
        f"  Turn {i+1}: {emotion[0]} (resolve to: {emotion[1]})"
        for i, emotion in enumerate(arc)
    )
    return GENERATION_PROMPT.format(
        num_turns=num_turns,
        description=scenario["description"],
        arc_str=arc_str,
    )


def build_system_prompt(num_turns: int) -> str:
    return SYSTEM_PROMPT.format(num_turns=num_turns)


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def call_ollama(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Call Ollama and return raw text, or None on failure."""
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            options={"temperature": GENERATION_TEMPERATURE},
        )
        return response["message"]["content"]
    except Exception as exc:
        log.warning(f"Ollama call failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# JSON parsing — robust extraction from LLM output
# ---------------------------------------------------------------------------

def extract_json(raw: str) -> Optional[dict]:
    """Extract the first JSON object from raw LLM output."""
    # Try direct parse first
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Try to find JSON block between { and final }
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try to find JSON block between ```json and ```
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    return None


def parse_conversation(raw: str, scenario: dict, conversation_id: str, num_turns: int) -> list[dict]:
    """Parse LLM output into a list of turn dicts.  Returns [] on failure."""
    data = extract_json(raw)
    if data is None:
        log.debug(f"JSON extraction failed for {conversation_id}")
        return []

    turns = data.get("turns")
    if not isinstance(turns, list) or len(turns) == 0:
        return []

    rows = []
    for turn in turns:
        tid = turn.get("turn_id")
        context = turn.get("context", "")
        text = turn.get("text", "").strip()
        raw_emotion = turn.get("emotion", "")

        # Validate
        if not text:
            continue
        emotion = normalize_emotion(raw_emotion)
        if emotion is None:
            log.debug(f"Unknown emotion '{raw_emotion}' in {conversation_id} turn {tid} — skipping turn")
            continue
        if len(text) < 5:
            continue

        rows.append({
            "conversation_id": conversation_id,
            "turn_id": int(tid) if tid else len(rows) + 1,
            "context": str(context).strip(),
            "text": text,
            "emotion": emotion,
            "scenario_category": scenario["category"],
        })

    return rows


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"conversation_count": 0, "emotion_counts": {e: 0 for e in CANONICAL_EMOTIONS}}


def save_checkpoint(conv_count: int, emotion_counts: dict):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"conversation_count": conv_count, "emotion_counts": emotion_counts}, f, indent=2)


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint()
    conversation_count = checkpoint["conversation_count"]
    emotion_counts = checkpoint["emotion_counts"]

    # Load existing data if resuming
    if OUTPUT_FILE.exists() and conversation_count > 0:
        df_existing = pd.read_csv(OUTPUT_FILE)
        all_rows = df_existing.to_dict("records")
        existing_texts = set(df_existing["text"].str.lower().str.strip().tolist())
        log.info(f"Resuming from checkpoint — {conversation_count} conversations, {len(all_rows)} turns so far")
    else:
        all_rows = []
        existing_texts = set()
        log.info("Starting fresh dataset generation")

    # Determine how many more conversations we need
    # We generate until all emotion targets are met
    def targets_met() -> bool:
        return all(emotion_counts.get(e, 0) >= EMOTION_TARGETS[e] for e in EMOTION_TARGETS)

    def remaining_emotions() -> list[str]:
        """Return emotions that still need more samples."""
        return [e for e in EMOTION_TARGETS if emotion_counts.get(e, 0) < EMOTION_TARGETS[e]]

    if targets_met():
        log.info("All emotion targets already met. Nothing to generate.")
        return

    # Shuffle scenarios to get good variety early
    random.seed(42 + conversation_count)
    scenarios_cycle = SCENARIOS.copy()

    # We need roughly (sum of remaining) / AVG_TURNS conversations
    total_remaining = sum(
        max(0, EMOTION_TARGETS[e] - emotion_counts.get(e, 0))
        for e in EMOTION_TARGETS
    )
    estimated_conversations = int(total_remaining / AVG_TURNS * 1.4)  # 40% buffer for failures

    log.info(f"Generating approximately {estimated_conversations} conversations "
             f"to reach {total_remaining} remaining turns")

    pbar = tqdm(total=TOTAL_TARGET, initial=sum(emotion_counts.values()), desc="Total turns")

    conv_idx = conversation_count
    since_checkpoint = 0

    for iteration in range(estimated_conversations * 10):  # generous outer limit
        if targets_met():
            break

        remaining = remaining_emotions()
        if not remaining:
            break

        # Pick a scenario whose arc covers at least one remaining emotion
        # Prefer scenarios that emit emotions we still need
        scenario_candidates = [
            s for s in scenarios_cycle
            if any(normalize_emotion(e[0]) in remaining for e in s["arc"])
        ]
        if not scenario_candidates:
            scenario_candidates = scenarios_cycle  # fallback: use all

        scenario = random.choice(scenario_candidates)

        # Decide number of turns: 2 or 3 (keep 4-turn conversations rare)
        num_turns = random.choices([2, 3, 3, 4], weights=[20, 40, 30, 10])[0]
        num_turns = min(num_turns, len(scenario["arc"]))

        conversation_id = f"C{conv_idx + 1:05d}"

        # --- Generate with retries ---
        rows = []
        for attempt in range(MAX_RETRIES):
            system_p = build_system_prompt(num_turns)
            user_p   = build_prompt(scenario, num_turns)
            raw      = call_ollama(system_p, user_p)
            if raw is None:
                time.sleep(1)
                continue
            rows = parse_conversation(raw, scenario, conversation_id, num_turns)
            if rows:
                break
            log.debug(f"Parse failed on attempt {attempt+1} for {conversation_id}")
            time.sleep(0.5)

        if not rows:
            log.warning(f"Skipping {conversation_id} after {MAX_RETRIES} failed attempts")
            continue

        # Deduplication — skip turns whose text already exists (case-insensitive)
        fresh_rows = []
        for row in rows:
            key = row["text"].lower().strip()
            if key not in existing_texts:
                existing_texts.add(key)
                fresh_rows.append(row)

        if not fresh_rows:
            continue

        # Only add turns for emotions that still need samples
        added = 0
        for row in fresh_rows:
            em = row["emotion"]
            if emotion_counts.get(em, 0) < EMOTION_TARGETS.get(em, 0):
                all_rows.append(row)
                emotion_counts[em] = emotion_counts.get(em, 0) + 1
                pbar.update(1)
                added += 1

        conv_idx += 1
        since_checkpoint += 1

        # Checkpoint periodically
        if since_checkpoint >= CHECKPOINT_EVERY:
            df = pd.DataFrame(all_rows)
            df.to_csv(OUTPUT_FILE, index=False)
            save_checkpoint(conv_idx, emotion_counts)
            since_checkpoint = 0
            log.info(f"Checkpoint saved — {conv_idx} conversations, {len(all_rows)} turns | "
                     + " | ".join(f"{e}:{emotion_counts.get(e,0)}" for e in CANONICAL_EMOTIONS))

    pbar.close()

    # Final save
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_FILE, index=False)
    save_checkpoint(conv_idx, emotion_counts)

    # ---------------------------------------------------------------------------
    # Summary report
    # ---------------------------------------------------------------------------
    log.info("=" * 60)
    log.info("DATASET GENERATION COMPLETE")
    log.info(f"Total conversations generated : {conv_idx}")
    log.info(f"Total turns (rows)            : {len(all_rows)}")
    log.info(f"Output file                   : {OUTPUT_FILE}")
    log.info("")
    log.info("Emotion distribution:")
    for emotion in CANONICAL_EMOTIONS:
        actual  = emotion_counts.get(emotion, 0)
        target  = EMOTION_TARGETS[emotion]
        status  = "✓" if actual >= target else f"SHORT by {target - actual}"
        log.info(f"  {emotion:<10} {actual:>5} / {target}  {status}")
    log.info("=" * 60)

    print("\n--- Dataset Generation Summary ---")
    print(f"Total turns generated: {len(all_rows)}")
    print(f"Output: {OUTPUT_FILE}")
    print("\nEmotion counts vs targets:")
    for emotion in CANONICAL_EMOTIONS:
        actual = emotion_counts.get(emotion, 0)
        target = EMOTION_TARGETS[emotion]
        bar = "#" * int(actual / target * 20)
        print(f"  {emotion:<10} {actual:>5}/{target}  [{bar:<20}]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info(f"Starting dataset generation with model: {OLLAMA_MODEL}")
    log.info(f"Target total: {TOTAL_TARGET} turns across {len(CANONICAL_EMOTIONS)} emotions")
    generate_dataset()
