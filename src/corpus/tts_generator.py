"""
Stage 4 of the corpus pipeline: turn a DOCTOR:/PATIENT: dialogue (.txt)
into a two-voice .mp3 with genuine speaker changes for M2 diarization.

Voices are assigned randomly per encounter (a different DOCTOR/PATIENT
pair per patient) for acoustic diversity across the corpus. Assignment is
seeded on the dialogue filename so the same input always yields the same
voices -- the corpus is reproducible.

Requires:
    pip install google-cloud-texttospeech pydub
    (pydub needs ffmpeg installed: `brew install ffmpeg` on macOS)
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
"""
import io
import os
import random

from google.cloud import texttospeech
from pydub import AudioSegment


# Pool of en-US voices to draw from. All same language so they blend
# naturally as one conversation; mixed genders/timbres so DOCTOR and
# PATIENT are acoustically distinct. Add or remove as you like.
VOICE_POOL = [
    "en-US-Neural2-A",   # male
    "en-US-Neural2-C",   # female
    "en-US-Neural2-D",   # male
    "en-US-Neural2-E",   # female
    "en-US-Neural2-F",   # female
    "en-US-Neural2-G",   # female
    "en-US-Neural2-I",   # male
    "en-US-Neural2-J",   # male
]


def parse_dialogue(text: str):
    """Split a DOCTOR:/PATIENT: dialogue into a list of (speaker, line) pairs."""
    turns = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("DOCTOR:"):
            turns.append(("DOCTOR", line[len("DOCTOR:"):].strip()))
        elif line.startswith("PATIENT:"):
            turns.append(("PATIENT", line[len("PATIENT:"):].strip()))
        else:
            # continuation of the previous turn -- append, don't drop
            if turns:
                turns[-1] = (turns[-1][0], turns[-1][1] + " " + line)
    return turns


def pick_voices(seed_key: str):
    """Pick two DISTINCT voices (one per speaker), seeded for reproducibility."""
    rng = random.Random(seed_key)              # seeded on the filename
    doctor_voice, patient_voice = rng.sample(VOICE_POOL, 2)  # sample() -> distinct
    return doctor_voice, patient_voice


def synthesize_dialogue(dialogue_path: str, output_path: str) -> None:
    """Read a dialogue .txt, synthesize each turn in its speaker's voice,
    concatenate with pydub, and write a single .mp3 to output_path."""

    # 1. READ the dialogue text.
    with open(dialogue_path, "r") as f:
        text = f.read()

    # 2. PARSE into (speaker, line) turns.
    turns = parse_dialogue(text)
    if not turns:
        raise ValueError(f"No DOCTOR:/PATIENT: turns found in {dialogue_path}")

    # Assign voices for THIS encounter, seeded on the filename so re-runs match.
    doctor_voice, patient_voice = pick_voices(os.path.basename(dialogue_path))

    # 3. SYNTHESIZE each turn with the right voice.
    client = texttospeech.TextToSpeechClient()
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    combined = AudioSegment.empty()            # start with silence
    for speaker, spoken in turns:
        voice_name = doctor_voice if speaker == "DOCTOR" else patient_voice
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name=voice_name,
        )
        response = client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=spoken),
            voice=voice,
            audio_config=audio_config,
        )

        # 4. CONCATENATE: decode this turn's MP3 bytes and append (pydub joins
        #    cleanly in decoded audio, so timestamps stay accurate for Whisper
        #    and pyannote -- unlike raw byte concatenation).
        turn_audio = AudioSegment.from_file(
            io.BytesIO(response.audio_content), format="mp3"
        )
        combined += turn_audio

    # 5. WRITE the combined audio to the output file.
    combined.export(output_path, format="mp3")



INPUT_DIR = "data/corpus"

def main():
    # Loop over all .txt files
    # txt_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))
    txt_files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".txt"))
    total = len(txt_files)                  
    print(f"Found {total} dialogues in {INPUT_DIR}\n")                                              
                                                                                                      
    generated = 0                                                                                   
    skipped = 0
    failed = 0                                                                                     
                                                                                                      
    for i, txt_path in enumerate(txt_files, start=1):
        txt_path = os.path.join(INPUT_DIR,txt_path)
        mp3_path = txt_path[:-4] + ".mp3"

        if os.path.exists(mp3_path):
            print(f"[{i}/{len(txt_files)}] skip (exist): {txt_path}")
            skipped += 1
            continue

        print(f"[{i}/{len(txt_files)}] synthesizing: {txt_path}")
        try:
            synthesize_dialogue(txt_path, mp3_path)
            generated  += 1
        except Exception as error:
            print(f"  FAILED: {error}")
            failed += 1

    print(f"\nDone, Generated: {generated} | Skipped: {skipped} | Failed: {failed}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()