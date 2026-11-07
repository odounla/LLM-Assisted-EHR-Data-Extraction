# M2 Audio Pipeline — Design Decisions

**Date:** Oct 21, 2026
**Status:** Ready to implement

---

## Goal

Take a two-voice `.mp3` encounter file and produce a structured JSON with labeled DOCTOR/PATIENT turns — ready for the FHIR generation stage.

---

## Tools

| Tool | Version | Purpose |
|---|---|---|
| Whisper | large-v3 | ASR — converts audio to timestamped transcript |
| pyannote.audio | 4.0.7 | Speaker diarization — identifies who spoke when |

**Note:** Whisper runs on CPU (FP32) — slower but functional for thesis evaluation. GPU would be needed for production speed.

---

## Pipeline Structure

```
src/pipeline/
    transcribe.py      — Whisper: mp3 → raw transcript with timestamps
    diarize.py         — pyannote: mp3 → speaker segments (SPEAKER_00, SPEAKER_01)
    align.py           — merge transcript + speaker labels → labeled turns
    audio_pipeline.py  — orchestrator: calls all three, writes output JSON
```

---

## Output Format

Each encounter produces one JSON file:

```json
{
  "encounter_id": "Abel832_Lubowitz58_...",
  "turns": [
    {"speaker": "DOCTOR", "text": "Good to see you again. How have you been feeling?"},
    {"speaker": "PATIENT", "text": "Honestly, pretty tired most days."},
    {"speaker": "DOCTOR", "text": "I hear you, and that fatigue is something I want to address."}
  ]
}
```

---

## Key Design Decisions

### 1. Speaker Assignment: First Speaker = DOCTOR
pyannote labels speakers as `SPEAKER_00` and `SPEAKER_01` — it does not know which is the doctor.

**Decision:** Map `SPEAKER_00` → `DOCTOR`, `SPEAKER_01` → `PATIENT`.

**Rationale:** The corpus was generated with the doctor always speaking first. This heuristic is reliable for this dataset.

**Limitation:** Would break on real clinic audio where a patient speaks first. A keyword-based classifier (medical terms → DOCTOR, patient-language → PATIENT) would be more robust for future work.

---

### 2. Merge Strategy: Align by Timestamp
Whisper produces word-level timestamps. pyannote produces speaker segments with start/end times. The `align.py` step assigns each Whisper word to the speaker active at that timestamp, then groups consecutive words from the same speaker into turns.

---

### 3. Model Loading
Whisper `large-v3` is loaded once per pipeline run — not once per file. Loading the model is expensive (~5 seconds); reusing it across 641 files saves significant time.

Same applies to the pyannote pipeline object.

---

### 4. Output Location
Transcribed JSON files are written to `data/transcripts/` with the same stem as the source `.mp3`:

```
data/corpus/Abel832_Lubowitz58_....mp3
→ data/transcripts/Abel832_Lubowitz58_....json
```

---

## Future Improvements (noted as thesis limitations)
- Keyword-based speaker classifier for real clinic audio
- GPU inference for production latency targets
- Confidence scores per turn for hallucination risk estimation
