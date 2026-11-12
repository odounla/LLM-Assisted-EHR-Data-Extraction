from pyannote.core import Segment


def align(whisper_result: dict, diarization) -> list:
    """Combine Whisper text with pyannote speaker labels into dialogue turns.

    Args:
        whisper_result: dict returned by transcribe() with word_timestamps=True;
            each segment has a 'words' list, where each word has
            'start', 'end' and 'word'.
        diarization: pyannote speaker segments returned by diarize()
            (who spoke from when to when).

    Returns:
        List of turns in order, e.g.
        [{"speaker": "DOCTOR", "text": "..."},
         {"speaker": "PATIENT", "text": "..."}]

    How it works:
        1. For each Whisper word, pick the pyannote speaker whose speech
           overlaps that word's time window the most.
        2. Label the first speaker heard in the audio as DOCTOR and the
           other as PATIENT (the corpus dialogues always start with the doctor).
        3. Merge consecutive words from the same speaker into one turn.

    Spacing: Whisper puts a leading space on a piece that starts a new word
    (" pre") and none on a piece that continues one ("-diabetes", ".85").
    Pieces are joined as-is, so "pre-diabetes" and "5.85%" stay intact.
    """
    turns = []
    speaker_map = {}  # maps SPEAKER_00/01 -> DOCTOR/PATIENT

    for seg in whisper_result["segments"]:
        for w in seg["words"]:
            start = w["start"]
            end = w["end"]
            text = w["word"]  # keep Whisper's own spacing (no .strip())

            # Step 1: find the dominant speaker for this word
            window = Segment(start, end)
            cropped = diarization.crop(window)
            if cropped.labels():
                raw_speaker = max(cropped.labels(), key=cropped.label_duration)
            elif turns:
                # word falls in a tiny gap between pyannote segments:
                # keep it with the current speaker instead of dropping it
                turns[-1]["text"] += text  # no added space
                continue
            else:
                continue

            # Step 2: first speaker heard = DOCTOR, other = PATIENT
            if raw_speaker not in speaker_map:
                speaker_map[raw_speaker] = "DOCTOR" if not speaker_map else "PATIENT"
            speaker = speaker_map[raw_speaker]

            # Step 3: merge with previous turn if same speaker
            if turns and turns[-1]["speaker"] == speaker:
                turns[-1]["text"] += text  # no added space
            else:
                # a new turn starts clean, without the leading space
                turns.append({"speaker": speaker, "text": text.strip()})

    return turns