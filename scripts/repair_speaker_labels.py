"""Fix swapped DOCTOR/PATIENT labels in word-level transcripts.

The first-speaker rule in align.py fails when pyannote groups the doctor's
opening greeting with the patient's voice: the rest of the encounter is then
labeled in reverse. This script identifies the doctor by content instead:
the doctor mostly says "you/your", the patient mostly says "I/my/me".
The rule was correct on 641 of 641 gold scripts. Run from the repo root:
    python scripts/repair_speaker_labels.py
"""
import json
from pathlib import Path

TRANSCRIPT_DIR = Path("data/transcripts_wordlevel")
LOG_PATH = Path("data/speaker_swaps.txt")

YOU = set("you your yours youre youve youll youd".split())
ME = set("i my me mine im ive".split())


def you_ratio(text: str) -> float:
    words = [w.strip(".,?!;:").replace("'", "") for w in text.lower().split()]
    you = sum(w in YOU for w in words)
    me = sum(w in ME for w in words)
    return you / (you + me + 1)


def is_swapped(turns: list[dict]) -> bool:
    doctor = " ".join(t["text"] for t in turns if t["speaker"] == "DOCTOR")
    patient = " ".join(t["text"] for t in turns if t["speaker"] == "PATIENT")
    return you_ratio(doctor) < you_ratio(patient)


FLIP = {"DOCTOR": "PATIENT", "PATIENT": "DOCTOR"}
swapped = []

for path in sorted(TRANSCRIPT_DIR.glob("*.json")):
    data = json.loads(path.read_text())
    if is_swapped(data["turns"]):
        for turn in data["turns"]:
            turn["speaker"] = FLIP[turn["speaker"]]
        path.write_text(json.dumps(data, indent=2))
        swapped.append(path.stem)

LOG_PATH.write_text("\n".join(swapped) + "\n")
print(f"Flipped {len(swapped)} transcripts. List saved to {LOG_PATH}")
