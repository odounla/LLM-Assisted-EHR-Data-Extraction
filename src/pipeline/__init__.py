

def __init__(self, model_name: str = "en_ner_bc5cdr_md"):                
      self.logger = logging.getLogger(__name__)                            
      self.nlp = spacy.load(model_name)                                    
      self.nlp.add_pipe("medspacy_context") 


if __name__ == "__main__":
    ner = NERPipeline()
    doc = ner.nlp("Your hemoglobin A1c came back at 5.85%. No history of anemia. Continue metformin.")
    for ent in doc.ents:
        print(ent.text, ent.label_, "negated:", ent._.is_negated, "historical:", ent._.is_historical)