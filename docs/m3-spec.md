# M3 Pipeline — Full Spec

**Date:** 2026-10-29
**Status:** In progress — `ner_pipeline.py`, `base_prompt.py`, `condition.py`, `medication_statement.py` spec complete. `observation.py` next.

---

## Open Issues — Resolved

| # | Issue | Resolution |
|---|---|---|
| 1 | Code assignment | LLM outputs name only. `umls_lookup.py` does post-LLM lookup. Failure rule: exact match → best-ranked → empty + flagged. Need UMLS UTS account before building. |
| 2 | Three conditions | `audio_only`, `multimodal`, `gold_transcript` — see table below |
| 3 | historical → inactive | Default to `active`. Use `note` field for "history of" language (verbatim only). `remission` maps to FHIR `remission`. **Clinician check required.** |
| 4 | Negation filtering scope | Filter before LLM for all resources except MEDICATION. Document in Evaluation chapter. FHIR `refuted` not used — documented scope decision. |

## Pending Clinician Check (ask both in one conversation)

1. `historical=true` → `clinicalStatus="active"` — is this the right default for chronic endocrinology conditions?
2. `"used to take [medication]"` → `status="completed"` — is `completed` correct, or `stopped`?

---

## Confirmed Decisions

- **NER:** Hybrid (scispaCy + medspaCy → LLM for FHIR structuring)
- **NER runs twice:** once on Whisper transcripts (`ner_output/whisper/`, shared by `audio_only` and `multimodal`), once on gold dialogue scripts (`ner_output/gold/`, used only by `gold_transcript`)
- **Code + display assignment:** UMLS/RxNorm API lookup post-LLM — LLM never generates codes or display text
- **LLM:** Parallel — GPT-4o (primary) + Llama 3 70B via Together.ai (baseline/comparison)
- **Three conditions:** `audio_only`, `multimodal`, `gold_transcript`
- **EHR context rule (all resource prompts):** EHR context confirms or adds detail to entities already mentioned in the encounter — never a source of new FHIR resources
- **CoT prompts:** One prompt per resource type (5 prompts per encounter)
- **Llama hosting:** Together.ai API
- **Negation filtering:** filter before LLM for all resource types except MEDICATION; document scope in Evaluation chapter
- **Speaker labels:** pyannote outputs `SPEAKER_00`/`SPEAKER_01` (two digits). Normalize to this format throughout.
- **Speaker label conversion for gold:** convert by first appearance — first speaker in the script becomes `SPEAKER_00`, second becomes `SPEAKER_01`. This matches pyannote's assignment logic. Do not map by role (DOCTOR/PATIENT), which would make role resolution artificially easy in the gold condition.
- **Speaker-role resolution:** done in `fhir_pipeline.py` before calling prompts, from dialogue patterns. Passed to prompts as resolved roles.

---

## Three Conditions

| Condition | Input | What it measures |
|---|---|---|
| `audio_only` | Whisper transcript + entities from `ner_output/whisper/` | Real pipeline from speech |
| `multimodal` | Whisper transcript + entities from `ner_output/whisper/` + Synthea FHIR context | Value added by prior EHR record |
| `gold_transcript` | Gold dialogue script + entities from `ner_output/gold/` | ASR error propagation — isolates speech recognition as the only variable |

**Why gold_transcript, not text_only:**
`text_only` can't capture anything from the current visit — score on new facts is near zero by design. `gold_transcript` vs `audio_only` directly isolates the ASR effect, which is a stated finding in the Methods chapter.

**Naming note:** In the thesis, describe `audio_only` as "the transcript from audio."

---

## File Structure

```
src/
├── pipeline/
│   ├── audio_pipeline.py        ← exists, complete (M2)
│   ├── ner_pipeline.py          ← NEW (--source whisper|gold)
│   ├── umls_lookup.py           ← NEW (post-LLM code + display assignment)
│   ├── fhir_pipeline.py         ← NEW (speaker-role resolution + LLM orchestration)
│   ├── fhir_validator.py        ← NEW
│   └── pipeline_runner.py       ← NEW
│
└── prompts/
    ├── __init__.py
    ├── base_prompt.py           ← NEW
    ├── condition.py             ← SPEC COMPLETE
    ├── medication_statement.py  ← SPEC COMPLETE
    ├── observation.py           ← next
    ├── care_plan.py
    └── procedure.py

data/pipeline/
├── ner_output/
│   ├── whisper/                 ← shared by audio_only and multimodal
│   └── gold/                    ← used only by gold_transcript
├── audio_only/
│   ├── fhir_gpt4o/
│   ├── fhir_llama/
│   ├── fhir_validated/
│   └── disagreements/
├── multimodal/
│   ├── fhir_gpt4o/
│   ├── fhir_llama/
│   ├── fhir_validated/
│   └── disagreements/
├── gold_transcript/
│   ├── fhir_gpt4o/
│   ├── fhir_llama/
│   ├── fhir_validated/
│   └── disagreements/
└── ablation/
    └── condition_diffs/         ← audio_only vs gold_transcript (ASR effect)
                                    audio_only vs multimodal (EHR context effect)
```

**What feeds M4 from M3:**

| Signal | Source |
|---|---|
| Model disagreement | `{condition}/disagreements/` across all three conditions |
| ASR error propagation | `ablation/condition_diffs/` — audio_only vs gold_transcript |
| EHR context value | `ablation/condition_diffs/` — audio_only vs multimodal |
| Ground truth mismatch | `fhir_validated/` vs Synthea FHIR |

---

## Build Order

1. `ner_pipeline.py` — entity extraction ← **ready to code**
2. `prompts/base_prompt.py` — context injection
3. `prompts/condition.py` — Condition resource
4. `prompts/medication_statement.py` — MedicationStatement
5. `prompts/observation.py` — next spec
6. `prompts/care_plan.py`
7. `prompts/procedure.py`
8. `fhir_pipeline.py` — speaker-role resolution + parallel LLM orchestration
9. `umls_lookup.py` — post-LLM code + display assignment
10. `fhir_validator.py` — HAPI + UMLS/RxNorm validation
11. `pipeline_runner.py` — batch orchestrator with skip/resume

**Pre-build action:** Set up UMLS UTS account (free) before building `umls_lookup.py`. RxNorm via RxNav API is open — no account needed.

---

## `src/pipeline/ner_pipeline.py` — SPEC COMPLETE

### Purpose
Extract structured clinical entities from transcripts. Runs twice: once on Whisper transcripts (output shared by `audio_only` and `multimodal`), once on gold dialogue scripts (output used only by `gold_transcript`).

### Source Flag
```python
def process_all(transcripts_dir, output_dir, source: str, force=False):
    # source: "whisper" | "gold"
    # skip if output exists and not force
    # log: processed | skipped | failed
```

### Speaker Label Normalization
pyannote outputs `SPEAKER_00`/`SPEAKER_01` (two digits). Normalize all speaker labels to this format at ingestion. For gold scripts: convert by first appearance — the first speaker in the script becomes `SPEAKER_00`, the second becomes `SPEAKER_01`. Do not map by role name. This matches pyannote's assignment logic and prevents the gold condition from having an artificial labeling advantage.

### Dependencies
```
scispacy
en_ner_bc5cdr_md      ← DISEASE → CONDITION, CHEMICAL → MEDICATION
medspacy              ← negation + assertion (NegEx context component)
```
`en_ner_bc5cdr_md` does not cover labs or vitals — that is why the custom `EntityRuler` exists.

### Input
Speaker-attributed turns JSON (normalized labels):
```json
[
  {"speaker": "SPEAKER_00", "text": "Your HbA1c came back at 8.2 percent.", "turn_index": 3},
  {"speaker": "SPEAKER_01", "text": "I have not been taking the metformin.", "turn_index": 4}
]
```

### Output
`data/pipeline/ner_output/whisper/{encounter_id}.json` or `data/pipeline/ner_output/gold/{encounter_id}.json`
```json
{
  "encounter_id": "Abel832_Lubowitz58_...",
  "source": "whisper",
  "entities": [
    {
      "text": "HbA1c", "label": "LAB",
      "negated": false, "uncertain": false, "historical": false, "hypothetical": false,
      "value": 8.2, "unit": "%",
      "speaker": "SPEAKER_00", "turn_index": 3
    },
    {
      "text": "metformin", "label": "MEDICATION",
      "negated": true, "uncertain": false, "historical": false, "hypothetical": false,
      "drug_dose": null, "drug_frequency": null,
      "speaker": "SPEAKER_01", "turn_index": 4
    }
  ]
}
```

Note: `value` is a **number**, not a string.

### Entity Schema

| Label | Attributes |
|---|---|
| `CONDITION` | text, negated, uncertain, historical, hypothetical, speaker, turn_index |
| `MEDICATION` | text, negated, uncertain, historical, hypothetical, drug_dose, drug_frequency, speaker, turn_index |
| `LAB` | text, negated, uncertain, historical, value (number), unit, speaker, turn_index |
| `VITAL` | text, negated, uncertain, historical, value (number), unit, speaker, turn_index |
| `PROCEDURE` | text, negated, uncertain, historical, hypothetical, speaker, turn_index |

### Processing Steps

**Step 1 — Model setup (once at init)**
- Load `en_ner_bc5cdr_md`
- Add medspaCy `context` component for negation/assertion/temporality
- Add custom `EntityRuler` before NER for endocrinology lab/vital patterns

**Step 2 — Custom EntityRuler patterns**

Labs: `HbA1c, hemoglobin A1c, A1C, fasting glucose, blood glucose, TSH, free T4, creatinine, eGFR, LDL, HDL, triglycerides`

Vitals: `blood pressure, systolic, diastolic, weight, BMI, heart rate, temperature`

Value extraction: regex `\d+\.?\d*` + unit (`%`, `mg/dL`, `mmol/L`, `mmHg`, `lbs`, `kg`) on token span following entity. Cast extracted value to `float`.

**Step 3 — Per-turn processing**
- Concatenate all turns, preserve `turn_index` and `speaker`
- Run: EntityRuler → NER → context
- Map: `DISEASE` → `CONDITION`, `CHEMICAL` → `MEDICATION`
- Map: `is_negated`, `is_uncertain`, `is_historical`, `is_hypothetical`

**Step 4 — Post-processing**
- MEDICATION: extract dose + frequency from 5-token window using regex
- Deduplicate on `(text, label, turn_index)`
- Filter `len(text) < 3`

**Step 5 — Output**
- Write JSON to output dir, skip if exists

### Acceptance Criteria
- Negation: `"patient denies chest pain"` → `chest pain`, `negated=true`
- Lab value: `"HbA1c came back at 8.2 percent"` → `label=LAB`, `value=8.2` (number), `unit="%"`
- Medication+dose: `"metformin 500mg twice daily"` → `drug_dose="500mg"`, `drug_frequency="twice daily"`
- Historical: `"history of hypertension"` → `historical=true`
- Skip/resume: output exists → skipped, logged
- F1 ≥ 0.85 on 50 held-out encounters vs Synthea ground truth

### Evaluation Note
Negated MEDICATION entities are passed to the LLM with `negated=true` — not filtered. For all other entity types, negated entities are filtered. Negation-error evaluation covers the NER stage for non-medication entities, and the LLM stage for medications. Document both scopes in Evaluation chapter.

---

## `src/prompts/base_prompt.py` — SPEC COMPLETE

### Interface
```python
def build_prompt(
    transcript: list[dict],
    entities: list[dict],
    resource_type: str,
    resource_schema: str,
    resource_instructions: str,
    ehr_context: dict | None,
    filter_negated: bool = True,    # False for medication_statement only
) -> list[dict]:
```

`transcript` and `entities` are always required — all three conditions provide both.

### Three Conditions

| Condition | transcript | entities source | ehr_context |
|---|---|---|---|
| `audio_only` | Whisper transcript | `ner_output/whisper/` | `None` |
| `multimodal` | Whisper transcript | `ner_output/whisper/` | Synthea bundle |
| `gold_transcript` | Gold dialogue script | `ner_output/gold/` | `None` |

### EHR Context Rule
EHR context confirms or adds detail to entities already mentioned in the encounter — never a source of new FHIR resources. Enforced in the system message and repeated in each resource prompt's CoT instructions.

### EHR Context Formatting
```python
def _format_ehr_context(bundle: dict) -> str:
    # extract only: existing conditions, active medications, recent labs
    # return as compact readable text — not raw FHIR JSON
    # omit everything else from the Synthea bundle
```
Passing the full Synthea bundle would change what multimodal measures and inflate token costs.

### System Message
```
You are a clinical documentation assistant specializing in FHIR R4 resource
generation for endocrinology outpatient encounters. Extract information from
the provided clinical data and generate valid FHIR R4 JSON.
Do not infer information not explicitly stated in the input.
Use EHR context only to add detail to resources derived from the encounter —
do not create new resources from the EHR context alone.
Do not assign terminology codes or display text — use "code_placeholder" for
all coding.code values. Codes and display text will be assigned by a separate
lookup step.
```

### User Message Sections (assembled conditionally)
```
## DIALOGUE TRANSCRIPT
SPEAKER_00 [turn 1]: ...
SPEAKER_01 [turn 2]: ...

## EXTRACTED ENTITIES
- "type 2 diabetes" | CONDITION | historical: false | speaker: SPEAKER_00 | turn: 3

## EHR CONTEXT              ← only if ehr_context is not None
{output of _format_ehr_context — existing conditions, active meds, recent labs only}

## FHIR R4 SCHEMA
{resource_schema}

## TASK
{resource_instructions}

## OUTPUT
Return a JSON array of FHIR R4 {resource_type} resources.
Output valid JSON only, no explanation, no markdown.
Use "code_placeholder" for all coding.code values.
```

### Key Rules
- `filter_negated=True`: filter negated entities before formatting (all resources except MEDICATION)
- `filter_negated=False`: include all entities with `negated` flag (MEDICATION only)
- `ehr_context=None` → no `## EHR CONTEXT` section (not empty)
- Transcript as flat text, not JSON
- LLM never outputs codes or display text

---

## `src/prompts/condition.py` — SPEC COMPLETE

### Interface
```python
def build(
    transcript: list[dict],
    entities: list[dict],
    ehr_context: dict | None = None,
) -> list[dict]:
```

### FHIR R4 Condition Schema
```json
{
  "resourceType": "Condition",
  "id": "<encounter_id>-condition-<n>",
  "clinicalStatus": {
    "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                "code": "<active | recurrence | relapse | inactive | remission | resolved>"}]
  },
  "verificationStatus": {
    "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                "code": "<confirmed | unconfirmed>"}]
  },
  "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-category",
                             "code": "encounter-diagnosis"}]}],
  "code": {
    "coding": [{"system": "http://snomed.info/sct", "code": "code_placeholder"}],
    "text": "<condition name as stated in dialogue>"
  },
  "subject": {"reference": "Patient/<patient_id>"},
  "note": [{"text": "<verbatim phrase from dialogue — only if present>"}]
}
```

### CoT Instructions
```
Generate one FHIR R4 Condition resource per condition explicitly mentioned
in this encounter. Think step by step:

1. Identify all conditions from the entity list. Use EHR context only to
   confirm or add detail to conditions already mentioned in the encounter —
   never as a source of new Condition resources.

2. For each condition:
   a. Determine clinicalStatus:
      - "active"    → default for any condition mentioned without resolution language
      - "remission" → only if dialogue explicitly says "in remission"
      - "resolved"  → only if dialogue explicitly says "resolved" or "cured"
      - "inactive"  → only if dialogue explicitly says "no longer active"
      Note: "history of X" does NOT mean inactive — default to active.
   b. Confirmed or uncertain?
      → confirmed:  verificationStatus = "confirmed"
      → uncertain:  verificationStatus = "unconfirmed"
   c. Use "code_placeholder" for coding.code. Do not generate codes or display text.
   d. If the dialogue contains exact language about the condition (e.g. "history
      of hypertension"), copy that phrase verbatim into the note field.
      Do not paraphrase or generate new text.

3. Do not include negated conditions — already excluded from entity list.
4. Do not infer conditions not explicitly stated in the encounter.
```

### Acceptance Criteria
- Active: `"type 2 diabetes"`, `historical=false` → `clinicalStatus="active"`
- Historical: `"history of hypertension"`, `historical=true` → `clinicalStatus="active"`, `note[0].text="history of hypertension"`
- Remission: dialogue says "in remission" → `clinicalStatus="remission"`
- Uncertain: `uncertain=true` → `verificationStatus="unconfirmed"`
- EHR context does not create new resources: condition in Synthea but not in dialogue → no resource generated
- No hallucination: no condition in input → `[]`
- No codes or display generated

### Pending
Confirm `historical → active` with ThyraHealth clinician.

---

## `src/prompts/medication_statement.py` — SPEC COMPLETE

### Key Deviation — Negation Handling
Negated MEDICATION entities are **not** filtered before the LLM (`filter_negated=False`). The `negated=true` flag is a hint for the LLM to resolve from transcript context — not an automatic mapping to `not-taken`.

### Interface
```python
def build(
    transcript: list[dict],
    entities: list[dict],         # includes negated MEDICATION entities
    ehr_context: dict | None = None,
) -> list[dict]:
```

### FHIR R4 MedicationStatement Schema
```json
{
  "resourceType": "MedicationStatement",
  "id": "<encounter_id>-medstmt-<n>",
  "status": "<active | completed | intended | stopped | not-taken | on-hold | unknown>",
  "medicationCodeableConcept": {
    "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "code_placeholder"}],
    "text": "<medication name as stated in dialogue>"
  },
  "subject": {"reference": "Patient/<patient_id>"},
  "informationSource": {"reference": "<Patient/<patient_id> | Practitioner/<practitioner_id>>"},
  "dosage": [{
    "text": "<free text dosage if mentioned>",
    "timing": {"repeat": {"frequency": 2, "period": 1, "periodUnit": "d"}},
    "doseAndRate": [{"doseQuantity": {
      "value": 500,
      "unit": "mg",
      "system": "http://unitsofmeasure.org",
      "code": "mg"
    }}]
  }]
}
```

Notes:
- `value`, `frequency`, `period` are **numeric, not strings**
- `display` omitted from `coding` — filled by `umls_lookup.py`
- `doseQuantity` requires both `unit` (human-readable) and `code` (UCUM) — FHIR validator checks both
- Insulin: use `U` or `[IU]`, not `"units"`

### CoT Instructions
```
Generate one FHIR R4 MedicationStatement per medication explicitly mentioned
in this encounter. Think step by step:

1. Identify all MEDICATION entities, including those with negated=true.
   Use EHR context only to confirm or add detail to medications already
   mentioned in the encounter — never as a source of new resources.

2. For each medication, determine status by reading the transcript context:
   - If negated=true: check what is negated in the transcript.
     → Patient is not taking the medication → status = "not-taken"
     → Negation applies to something else (side effects, allergies, etc.)
        and patient is taking the medication → status = "active"
     → Medication mentioned only in negated context and patient is not
        taking it (e.g. "no allergy to penicillin") → generate no
        MedicationStatement for this entity
   - Explicitly stopped → status = "stopped"
   - Newly prescribed this visit → status = "intended"
   - Currently taking, confirmed → status = "active"
   - "used to take" / "was on" → status = "completed"
   - On hold per clinician → status = "on-hold"
   - Historical mention, unclear if current → status = "unknown"

3. Determine informationSource from resolved speaker roles:
   - Patient reports → "Patient/<patient_id>"
   - Clinician prescribes/confirms → "Practitioner/<practitioner_id>"

4. Populate dosage from entity attributes:
   - If drug_dose present: populate doseQuantity (numeric value, unit, UCUM code)
   - If drug_frequency present: populate timing.repeat (numeric values)
   - Omit dosage entirely only if both drug_dose and drug_frequency are null
   - If only one is present, include only that subfield

5. Use "code_placeholder". Do not generate display text.
6. Do not infer medications not explicitly stated in the input.
```

### Status Mapping Reference

| Dialogue language | FHIR status |
|---|---|
| "I have not been taking..." | `not-taken` |
| "No side effects from [medication]" | `active` |
| "No allergy to [medication]", not taking it | no MedicationStatement |
| "I stopped taking..." | `stopped` |
| "I am prescribing..." / "start taking..." | `intended` |
| "I am taking..." / "she takes..." | `active` |
| "used to take..." / "was on..." | `completed` |
| "holding the medication" | `on-hold` |
| Historical mention, unclear if current | `unknown` |

### Frequency Mapping Reference

| Dialogue language | timing.repeat |
|---|---|
| "once daily" | frequency=1, period=1, periodUnit="d" |
| "twice daily" / "BID" | frequency=2, period=1, periodUnit="d" |
| "three times daily" / "TID" | frequency=3, period=1, periodUnit="d" |
| "once weekly" | frequency=1, period=1, periodUnit="wk" |
| "every 12 hours" | frequency=1, period=12, periodUnit="h" |

### UCUM Unit Reference

| Drug unit | UCUM code |
|---|---|
| milligrams | `mg` |
| micrograms | `ug` |
| millilitres | `mL` |
| insulin units | `U` or `[IU]` — not "units" |

### Acceptance Criteria
- Active with dose: `negated=false`, `drug_dose="500mg"`, `drug_frequency="twice daily"` → `status="active"`, `value=500` (numeric), `code="mg"`, `frequency=2`
- Not-taken: `negated=true`, transcript "I have not been taking my metformin" → `status="not-taken"`, entity not dropped
- Negation on side effects: `negated=true`, transcript "no side effects from metformin" → `status="active"`
- No MedicationStatement: `negated=true`, transcript "no allergy to penicillin" → no resource generated
- Stopped: "I stopped taking lisinopril" → `status="stopped"`
- Completed: "she used to take metformin" → `status="completed"`
- Intended: "I am going to start you on semaglutide" → `status="intended"`
- Partial dosage — dose only: `drug_dose="50mcg"`, `drug_frequency=null` → `doseQuantity` present (`value=50`, `unit="ug"`, `code="ug"`), `timing` omitted
- No dosage: both null → `dosage` omitted entirely
- EHR context does not create new resources
- No hallucination → `[]`
- No codes or display generated

### Pending
Confirm `"used to take" → completed` vs `stopped` with ThyraHealth clinician.

### Methods Chapter Note
`status="intended"` for new prescriptions is a documented simplification — full FHIR uses `MedicationRequest`. Note in Methods.

---

## `src/pipeline/umls_lookup.py` — SPEC TODO

### Purpose
Post-LLM. Replaces `"code_placeholder"` with authoritative codes and fills `display` text.

### Lookup Failure Rule
1. Exact name match → use code and display
2. No exact match → use best-ranked match
3. No match → leave `code` empty, set `_code_status: "unresolved"`, count in metrics

### APIs
- **Conditions/Procedures:** UMLS API — requires free UTS account
- **Medications:** RxNorm via RxNav API — open, no account needed

---

## `src/prompts/observation.py` — TODO

## `src/prompts/care_plan.py` — TODO

## `src/prompts/procedure.py` — TODO

## `src/pipeline/fhir_pipeline.py` — TODO

### Known Responsibility
Speaker-role resolution before calling prompts. Determine which `SPEAKER_00`/`SPEAKER_01` is Practitioner and which is Patient from dialogue patterns (the speaker who orders, prescribes, and asks diagnostic questions is the Practitioner). Pass resolved roles to all resource prompts for `informationSource`.

## `src/pipeline/fhir_validator.py` — TODO

## `src/pipeline/pipeline_runner.py` — TODO
