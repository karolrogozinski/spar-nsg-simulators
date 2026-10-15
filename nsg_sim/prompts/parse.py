"""Read YES / NO from a simulator response."""

from __future__ import annotations

import re

# Answer section headers seen in practice: "[ANSWER]" (requested), "**ANSWER**" (llama3),
# "**Answer:**", "## Answer", "ANSWER:" at the start of a line. The value follows on the
# same line or the next one.
ANSWER_HEADER = re.compile(
    r"(?:\[\s*ANSWER\s*\]|\*\*\s*\[?\s*ANSWER\s*\]?\s*:?\s*\*\*|^[ \t]*#+[ \t]*ANSWER\b|^[ \t]*ANSWER[ \t]*:)"
    r"[ \t]*:?\s*(.+)",
    re.IGNORECASE | re.MULTILINE,
)


def parse_answer(text: str | None) -> str | None:
    """YES / NO, or None if missing or hedged (e.g. "YES/NO").

    Reasoning blocks (<think>...</think>) are dropped and the last answer section wins,
    since reasoning models may draft an answer before the final one.
    """
    text = re.sub(r"<think>.*?(</think>|$)", "", text or "", flags=re.DOTALL)
    matches = ANSWER_HEADER.findall(text)
    if not matches:
        # no answer header: accept a bare YES / NO as the first line (e.g. "YES\n\nCONFIDENCE\nMEDIUM")
        first = next((line for line in text.splitlines() if line.strip()), "")
        bare = first.strip().strip("*.:# ").upper()
        return bare if bare in ("YES", "NO") else None
    answer = matches[-1].strip().upper().strip("*.: ")
    if "YES" in answer and "NO" in answer.replace("NOT", ""):
        return None
    for valid in ("YES", "NO"):
        if answer.startswith(valid):
            return valid
    return None
