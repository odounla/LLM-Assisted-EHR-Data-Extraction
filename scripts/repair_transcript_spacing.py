"""Repair spacing artifacts in word-level transcripts.

align.py joined Whisper word pieces with a space, which split
hyphenated words ("pre -diabetes") and decimals ("5 .7").
This rejoins them. Run from the repo root:
    python scripts/repair_transcript_spacing.py
"""
import json
import re
from pathlib import Path

TRANSCRIPT_DIR = Path("data/transcripts_wordlevel")

RULES = [
    ("hyphen",   re.compile(r"(\w) -(\w)"),    r"\1-\2"),   # pre -diabetes -> pre-diabetes
    ("decimal",  re.compile(r"(\d) \.(\d)"),   r"\1.\2"),   # 5 .7 -> 5.7
    ("thousand", re.compile(r"(\d) ,(\d{3})"), r"\1,\2"),   # 1 ,000 -> 1,000
    ("percent",  re.compile(r"(\d) %"),        r"\1%"),     # 5.7 % -> 5.7%
    ("slash",    re.compile(r"(\d) /(\d)"),    r"\1/\2"),   # 120 /80 -> 120/80
]

counts = {name: 0 for name, _, _ in RULES}
files_changed = 0

for path in sorted(TRANSCRIPT_DIR.glob("*.json")):
    data = json.loads(path.read_text())
    changed = False
    for turn in data["turns"]:
        text = turn["text"]
        for name, pattern, replacement in RULES:
            # Loop until stable: "day -to -day" needs two passes.
            while True:
                text, n = pattern.subn(replacement, text)
                if n == 0:
                    break
                counts[name] += n
        if text != turn["text"]:
            turn["text"] = text
            changed = True
    if changed:
        path.write_text(json.dumps(data, indent=2))
        files_changed += 1

print("Files changed:", files_changed)
print("Fixes by rule:", counts)
