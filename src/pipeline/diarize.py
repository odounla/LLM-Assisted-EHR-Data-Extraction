import os

import torch
import torchaudio
from dotenv import load_dotenv
from pyannote.audio import Pipeline

load_dotenv()

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=os.getenv("HF_TOKEN"),
)


def diarize(audio_path: str) -> object:
    """Run pyannote speaker diarization on one .mp3.

    Returns an Annotation object with (start, end, speaker) segments.
    pyannote auto-converts stereo to mono and resamples to 16kHz.
    num_speakers=2 is hardcoded: one doctor, one patient per encounter.
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    result = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=2)
    return result