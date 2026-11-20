"""
Smoke test for dataset generation.
Generates exactly 20 conversations, prints results, and exits.
Used to verify the pipeline before the full overnight run.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.generate_dataset import (
    SCENARIOS,
    OLLAMA_MODEL,
    build_system_prompt,
    build_prompt,
    call_ollama,
    parse_conversation,
    normalize_emotion,
    CANONICAL_EMOTIONS,
)
import random

random.seed(99)

SMOKE_CONVERSATIONS = 20

print(f"Smoke test: generating {SMOKE_CONVERSATIONS} conversations with {OLLAMA_MODEL}")
print("-" * 60)

results = []
failures = 0

for i in range(SMOKE_CONVERSATIONS):
    scenario = random.choice(SCENARIOS)
    num_turns = random.choice([2, 3])
    conv_id = f"SMOKE_{i+1:03d}"

    system_p = build_system_prompt(num_turns)
    user_p   = build_prompt(scenario, num_turns)
    raw      = call_ollama(system_p, user_p)

    if raw is None:
        print(f"  [{i+1:2d}] FAIL — Ollama returned None")
        failures += 1
        continue

    rows = parse_conversation(raw, scenario, conv_id, num_turns)
    if not rows:
        print(f"  [{i+1:2d}] FAIL — JSON parse failed")
        print(f"         Raw output snippet: {raw[:200]}")
        failures += 1
        continue

    results.extend(rows)
    emotions = [r["emotion"] for r in rows]
    print(f"  [{i+1:2d}] OK  {conv_id}  turns={len(rows)}  emotions={emotions}")
    for r in rows:
        print(f"         turn {r['turn_id']}: [{r['emotion']:<8}] {r['text'][:80]}")

print("-" * 60)
print(f"\nTotal rows generated : {len(results)}")
print(f"Failed conversations : {failures} / {SMOKE_CONVERSATIONS}")

if results:
    import pandas as pd
    from collections import Counter
    df = pd.DataFrame(results)
    print(f"\nEmotion distribution in smoke test:")
    for em, cnt in Counter(df["emotion"]).most_common():
        print(f"  {em:<10} {cnt}")
    print(f"\nSample fields present: {list(df.columns)}")

success_rate = (SMOKE_CONVERSATIONS - failures) / SMOKE_CONVERSATIONS * 100
print(f"\nSuccess rate: {success_rate:.0f}%")
if success_rate >= 70:
    print("✓ Smoke test PASSED — safe to run full generation")
else:
    print("✗ Smoke test FAILED — check errors above before running full generation")
