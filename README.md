# LLM-Assisted EHR Data Extraction

MS Thesis — University of Washington
**Multimodal Clinical Documentation: LLM-Assisted EHR Data Extraction from Voice and Text for Endocrinology Outpatient Encounters**

## Overview

An end-to-end pipeline that takes ambient audio of physician-patient encounters and prior FHIR records as input, and outputs validated FHIR R4 structured documentation.

## Pipeline

1. **Audio Processing** — Whisper large-v3 (ASR) + pyannote.audio (speaker diarization)
2. **Clinical NLP** — Named entity extraction with negation and temporal handling
3. **FHIR Generation** — GPT-4o + Llama 3 70B with Chain-of-Thought prompting
4. **Consistency Checking** — Cross-modal consistency between audio and text context

## Project Structure

src/
├── corpus/ # Synthetic corpus generation (Synthea → TTS → Whisper)
├── pipeline/ # NER + FHIR generation
├── consistency/ # Cross-modal consistency checker
└── evaluation/ # Metrics and hallucination taxonomy scoring
data/              # Local only — not tracked by git
notebooks/         # Exploration and analysis
tests/ # Unit and integration tests

## Status

In progress — Summer 2026
