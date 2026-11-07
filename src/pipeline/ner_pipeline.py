import json
import logging
import re
from pathlib import Path
from typing import Literal

import spacy
import medspacy


class NERPipeline:

    def __init__(self, model_name: str = "en_ner_bc5cdr_md"):
        self.logger = logging.getLogger(__name__)
        self.nlp = spacy.load(model_name)
        self.nlp.add_pipe("medspacy_context")

    def process_encounter(self, turns: list[dict], encounter_id: str, source: Literal["whisper", "gold"]) -> dict:
        text = "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)
        doc = self.nlp(text)
        entities = [
             {
                  "text": ent.text,
                  "label": ent.label_,
                  "negated": ent._.is_negated,
                  "uncertain": ent._.is_uncertain,
                  "historical": ent._.is_historical

             }
             for ent in doc.ents
        ]
        return {"encounter_id": encounter_id, "source": source, "entities": entities}

    def process_all(self, input_dir: Path, output_dir: Path, source: Literal["whisper", "gold"], force: bool = False) -> None:
        pass



if __name__ == "__main__":                                             
      ner = NERPipeline()                                                  
      turns = [                               
          {"speaker": "SPEAKER_00", "text": "Your hemoglobin A1c came back at 5.85%."},                                                             
          {"speaker": "SPEAKER_01", "text": "No history of anemia. I continue taking metformin."},                                            
      ]                                                                    
      result = ner.process_encounter(turns, encounter_id="test_001", source="whisper")                                                        
      print(json.dumps(result, indent=2))   