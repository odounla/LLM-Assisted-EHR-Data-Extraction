"""
M2 orchestrator: run every .mp3 in data/corpus/ through
transcribe -> diarize -> align, and save labeled dialogue turns
as JSON in data/transcripts/.

Usage (from the repo root):
    python -m src.pipeline.audio_pipeline
"""
import json
import os

from src.pipeline.transcribe import transcribe
from src.pipeline.diarize import diarize
from src.pipeline.align import align

INPUT_DIR = "data/corpus"
OUTPUT_DIR = "data/transcripts_wordlevel"

# Cap how many files to process in one run. Keep it small while testing
# (~3-4 min per file on CPU). Set to None to run all of them.
LIMIT = 1


def process_one(mp3_path: str) -> dict:
    """Run one encounter through the full M2 pipeline."""
    whisper_result = transcribe(mp3_path)
    diarization = diarize(mp3_path)

    # pyannote 4.x may wrap the Annotation; unwrap it if so.
    diarization = getattr(diarization, "speaker_diarization", diarization)

    turns = align(whisper_result, diarization)

    encounter_id = os.path.basename(mp3_path)[:-4]
    return {"encounter_id": encounter_id, "turns": turns}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mp3_files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".mp3"))
    total = len(mp3_files)
    print(f"Found {total} audio files in {INPUT_DIR}\n")

    processed, skipped, failed = 0, 0, 0

    for i, filename in enumerate(mp3_files, start=1):
        mp3_path = os.path.join(INPUT_DIR, filename)
        json_path = os.path.join(OUTPUT_DIR, filename[:-4] + ".json")

        if os.path.exists(json_path):
            print(f"[{i}/{total}] skip (exists): {filename}")
            skipped += 1
            continue

        print(f"[{i}/{total}] processing: {filename}")
        try:
            result = process_one(mp3_path)
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)
            processed += 1
            print(f"    saved {len(result['turns'])} turns")
        except Exception as error:
            print(f"    FAILED: {error}")
            failed += 1

        if LIMIT and (processed + failed) >= LIMIT:
            print(f"\nReached LIMIT of {LIMIT}, stopping early.")
            break

    print(f"\nDone. Processed: {processed} | Skipped: {skipped} | Failed: {failed}")


if __name__ == "__main__":
    main()