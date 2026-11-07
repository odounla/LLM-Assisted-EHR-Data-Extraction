# Thesis Progress Update — October 30, 2026

**Multimodal Clinical Documentation: LLM-Assisted EHR Data Extraction**
Olivier Dounla Woumpe | MS CS&SE, University of Washington

---

## 1. Current Status

### M1: Corpus Generation — **Complete**

- 1,389 Synthea patients generated (seed 42, Massachusetts)
- **641 diabetic encounter dialogues** (.txt) — more than 3× the 200+ target
- **641 two-voice .mp3 files** via Google Cloud TTS (seeded for reproducibility)

### M2: Audio-Only Pipeline (Whisper + pyannote) — **In Progress**

**Infrastructure built:**
- `transcribe.py` — Whisper large-v3 transcription with segment timestamps
- `diarize.py` — pyannote 3.1 speaker diarization (2 speakers)
- `align.py` — matches transcript segments to speakers, labels DOCTOR/PATIENT, merges into turns
- `audio_pipeline.py` — batch orchestrator with resume logic, outputs JSON per encounter

**Milestones hit:**
- Whisper large-v3 and pyannote 3.1 installed and tested
- Fixed pyannote MP3 decoding error (load audio as waveform first)
- First end-to-end test run on 3 files: complete

**Blocker:** Whisper on CPU ~3.5 min/file → ~43+ hours for full 641-encounter corpus

---

## 2. Plan for Next Week

| Priority | Task |
|---|---|
| 1 | Run M2 on all 641 encounters (request GPU access; fall back to overnight local runs) |
| 2 | Validate M2 output on a small subset — compare DOCTOR/PATIENT turns against source dialogues |
| 3 | Begin M3: LLM orchestration → FHIR R4 generation |

---

## 3. Next Milestones

| Milestone | Description |
|---|---|
| **M3** | LLM orchestration → FHIR R4 generation (5 resource types) |
| **M4** | Cross-modal consistency checker |

---

## Other Notes

- **Graduate Research Mini-Grant:** Application link received from Jerry (Oct 19); application already submitted during last week of summer.
