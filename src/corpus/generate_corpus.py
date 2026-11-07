"""
Batch corpus generation: run the FHIR -> dialogue pipeline over every
diabetic patient bundle in a folder, saving each dialogue to data/corpus/.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...     (or OPENAI_API_KEY for the OpenAI path)
    python src/corpus/generate_corpus.py
"""
import os
import glob
import json

from src.corpus.fhir_extractor import extract_clinical_context
from src.corpus.dialogue_generator import generate_dialogue_anthropic

# Where the input bundles live and where dialogues get written.
INPUT_DIR = "output/fhir" 
OUTPUT_DIR = "data/corpus"

# Cap how many dialogues to generate in one run. Useful while testing so you
# don't spend an API call on every patient. Set to None to run all of them.
LIMIT = None


def is_patient_bundle(bundle: dict) -> bool:
    """Requirement 2: a real patient bundle contains a Patient resource.
    Content check, not filename check -- more robust."""
    return any(
        e.get("resource", {}).get("resourceType") == "Patient"
        for e in bundle.get("entry", [])
    )


def is_diabetic(context: dict) -> bool:
    """Requirement 3: a patient is diabetic if they have a diabetes-related
    condition on their problem list. Diabetes is a diagnosis, not a lab --
    a stray glucose value doesn't qualify. 'diab' catches Diabetes /
    Prediabetes / diabetic in one substring."""
    return any("diab" in c.lower() for c in context["conditions"])


def main():
    # Requirement 4 (prep): make sure the output folder exists.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Requirement 1: find every JSON bundle in the input folder.
    bundle_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    total = len(bundle_paths)
    print(f"Found {total} JSON files in {INPUT_DIR}\n")

    generated = 0
    skipped = 0

    for i, path in enumerate(bundle_paths, start=1):
        source_name = os.path.basename(path)

        # Load the file. If it's malformed, don't let it kill the whole batch.
        try:
            with open(path) as f:
                bundle = json.load(f)
        except (json.JSONDecodeError, OSError) as err:
            print(f"[{i}/{total}] {source_name} ... SKIP (could not read: {err})")
            skipped += 1
            continue

        # Requirement 2: skip non-patient bundles (hospital, practitioner).
        if not is_patient_bundle(bundle):
            print(f"[{i}/{total}] {source_name} ... skip (not a patient bundle)")
            skipped += 1
            continue

        # Stage 1: extract clinical context.
        context = extract_clinical_context(bundle)

        # Requirement 3: skip non-diabetic patients.
        if not is_diabetic(context):
            print(f"[{i}/{total}] {source_name} ... skip (not diabetic)")
            skipped += 1
            continue

        # Stage 2: generate dialogue. Wrap the API call so one failure
        # (network, rate limit) doesn't lose the 39 patients around it.
        out_name = source_name.replace(".json", ".txt")                                                     
        out_path = os.path.join(OUTPUT_DIR, out_name)                                                       
        if os.path.exists(out_path):                                                                        
            print(f"[{i}/{total}] {source_name} ... skip (already exists)")                                 
            continue  
        print(f"[{i}/{total}] {source_name} ... generating")
        try:
            dialogue = generate_dialogue_anthropic(context)
        except Exception as err:
            print(f"[{i}/{total}] {source_name} ... FAILED ({err})")
            skipped += 1
            continue

        # Requirement 4: save with same stem, .txt extension (traceable).
        with open(out_path, "w") as f:
            f.write(dialogue)

        generated += 1

        # Stop early once we hit the cap (skips are not counted toward LIMIT).
        if LIMIT and generated >= LIMIT:
            print(f"\nReached LIMIT of {LIMIT} -- stopping early.")
            break

    # Final summary so you know what happened without scrolling.
    print(f"\nDone. Generated {generated} dialogues, skipped {skipped}. "
          f"Output in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()