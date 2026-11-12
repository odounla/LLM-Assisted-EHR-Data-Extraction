import json
import logging
import re
from pathlib import Path
from typing import Literal

import spacy
import medspacy  # registers the "medspacy_context" component


class NERPipeline:

    def __init__(self, model_name: str = "en_ner_bc5cdr_md"):
        self.logger = logging.getLogger(__name__)
        self.nlp = spacy.load(model_name)

        # Labs and vitals: the model only knows DISEASE and CHEMICAL.
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns([
            {"label": "LAB", "pattern": [{"LOWER": "hba1c"}]},
            {"label": "LAB", "pattern": [{"LOWER": "a1c"}]},
            {"label": "LAB", "pattern": [{"LOWER": "hemoglobin"}, {"LOWER": "a1c"}]},
            {"label": "LAB", "pattern": [{"LOWER": "glucose"}]},
            {"label": "LAB", "pattern": [{"LOWER": "tsh"}]},
            {"label": "LAB", "pattern": [{"LOWER": "free"}, {"LOWER": "t4"}]},
            {"label": "LAB", "pattern": [{"LOWER": "creatinine"}]},
            {"label": "LAB", "pattern": [{"LOWER": "egfr"}]},
            {"label": "LAB", "pattern": [{"LOWER": {"IN": ["ldl", "hdl"]}}]},
            {"label": "LAB", "pattern": [{"LOWER": "triglycerides"}]},
            {"label": "VITAL", "pattern": [{"LOWER": "blood"}, {"LOWER": "pressure"}]},
            {"label": "VITAL", "pattern": [{"LOWER": "weight"}]},
            {"label": "VITAL", "pattern": [{"LOWER": "bmi"}]},
            {"label": "VITAL", "pattern": [{"LOWER": "heart"}, {"LOWER": "rate"}]},
        ])

        # Negation / uncertainty / historical / hypothetical flags.
        self.nlp.add_pipe("medspacy_context")

    def process_encounter(self, turns: list[dict], encounter_id: str,
                          source: Literal["whisper", "gold"]) -> dict:
        entities = []
        for turn_index, turn in enumerate(turns):
            # Whisper writes "pre-diabetes"; the tokenizer splits on the hyphen.
            text = re.sub(r"\bpre[- ]diabet", "prediabet", turn["text"], flags=re.IGNORECASE)
            doc = self.nlp(text)
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "speaker": turn["speaker"],
                    "turn_index": turn_index,
                    "negated": ent._.is_negated,
                    "uncertain": ent._.is_uncertain,
                    "historical": ent._.is_historical,
                    "hypothetical": ent._.is_hypothetical,
                })
        return {"encounter_id": encounter_id, "source": source, "entities": entities}

    def process_all(self, input_dir, output_dir,
                    source: Literal["whisper", "gold"], force: bool = False) -> None:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(input_dir.glob("*.json"))
        total = len(files)
        print(f"Found {total} transcripts in {input_dir}\n")

        processed, skipped, failed = 0, 0, 0
        for i, path in enumerate(files, start=1):
            out_path = output_dir / path.name
            if out_path.exists() and not force:
                skipped += 1
                continue
            try:
                data = json.loads(path.read_text())
                result = self.process_encounter(data["turns"], data["encounter_id"], source)
                out_path.write_text(json.dumps(result, indent=2))
                processed += 1
                print(f"[{i}/{total}] {path.name}: {len(result['entities'])} entities")
            except Exception as error:
                print(f"[{i}/{total}] {path.name}: FAILED ({error})")
                failed += 1

        print(f"\nDone. Processed: {processed} | Skipped: {skipped} | Failed: {failed}")


if __name__ == "__main__":
    ner = NERPipeline()

    # Quick check: should show a LAB, a negated DISEASE and a CHEMICAL.
    test = [{"speaker": "DOCTOR",
             "text": "Your hemoglobin A1c came back at 5.85%. No history of anemia. Continue metformin."}]
    for e in ner.process_encounter(test, "test", "whisper")["entities"]:
        print(e["text"], e["label"], "negated:", e["negated"])
