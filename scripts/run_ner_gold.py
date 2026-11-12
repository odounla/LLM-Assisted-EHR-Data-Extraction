"""Run NER on the gold dialogue scripts (data/corpus/*.txt).

Produces the same output format as the Whisper run, with source="gold",
for the gold_transcript condition. Run from the repo root:
    python -m scripts.run_ner_gold
"""
import json
from pathlib import Path

from src.pipeline.ner_pipeline import NERPipeline

INPUT_DIR = Path("data/corpus")
OUTPUT_DIR = Path("data/ner_output/gold")


def parse_dialogue(text: str) -> list[dict]:
    """Turn 'DOCTOR: ...' / 'PATIENT: ...' lines into a list of turns."""
    turns = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("DOCTOR:"):
            turns.append({"speaker": "DOCTOR", "text": line[len("DOCTOR:"):].strip()})
        elif line.startswith("PATIENT:"):
            turns.append({"speaker": "PATIENT", "text": line[len("PATIENT:"):].strip()})
        elif turns:
            # continuation of the previous turn
            turns[-1]["text"] += " " + line
    return turns


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ner = NERPipeline()

    files = sorted(INPUT_DIR.glob("*.txt"))
    total = len(files)
    print(f"Found {total} gold scripts in {INPUT_DIR}\n")

    processed, failed, empty = 0, 0, 0
    for i, path in enumerate(files, start=1):
        try:
            turns = parse_dialogue(path.read_text())
            if not turns:
                print(f"[{i}/{total}] {path.name}: NO TURNS FOUND")
                empty += 1
                continue
            result = ner.process_encounter(turns, path.stem, "gold")
            (OUTPUT_DIR / f"{path.stem}.json").write_text(json.dumps(result, indent=2))
            processed += 1
        except Exception as error:
            print(f"[{i}/{total}] {path.name}: FAILED ({error})")
            failed += 1

    print(f"Done. Processed: {processed} | Empty: {empty} | Failed: {failed}")


if __name__ == "__main__":
    main()
