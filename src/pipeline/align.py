from pyannote.core import Segment

YOU_WORDS = set("you your yours youre youve youll youd".split())
ME_WORDS = set("i my me mine im ive".split())


def _you_ratio(text: str) -> float:
    """Share of second-person words among second- and first-person words."""
    words = [w.strip(".,?!;:").replace("'", "") for w in text.lower().split()]
    you = sum(w in YOU_WORDS for w in words)
    me = sum(w in ME_WORDS for w in words)
    return you / (you + me + 1)


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
        2. Merge consecutive words from the same speaker into one turn.
        3. Decide which speaker is the doctor from what was said: the doctor
           mostly says "you/your", the patient mostly says "I/my/me".
           (The earlier rule, "first speaker heard = DOCTOR", labeled 145 of
           641 encounters in reverse, because pyannote sometimes groups the
           opening greeting with the patient's voice.)

    Spacing: Whisper puts a leading space on a piece that starts a new word
    (" pre") and none on a piece that continues one ("-diabetes", ".85").
    Pieces are joined as-is, so "pre-diabetes" and "5.85%" stay intact.
    """
    turns = []  # built with pyannote's raw labels first (SPEAKER_00, ...)

    for seg in whisper_result["segments"]:
        for w in seg["words"]:
            text = w["word"]  # keep Whisper's own spacing

            # Step 1: find the dominant speaker for this word
            cropped = diarization.crop(Segment(w["start"], w["end"]))
            if cropped.labels():
                raw_speaker = max(cropped.labels(), key=cropped.label_duration)
            elif turns:
                # word falls in a tiny gap between pyannote segments:
                # keep it with the current speaker instead of dropping it
                turns[-1]["text"] += text
                continue
            else:
                continue

            # Step 2: merge with previous turn if same speaker
            if turns and turns[-1]["speaker"] == raw_speaker:
                turns[-1]["text"] += text
            else:
                turns.append({"speaker": raw_speaker, "text": text.strip()})

    if not turns:
        return turns

    # Step 3: the raw speaker with the highest "you" ratio is the doctor.
    # Speakers are listed in order of first appearance, so on a tie the
    # first speaker heard is the doctor.
    raw_speakers = list(dict.fromkeys(t["speaker"] for t in turns))
    ratios = {
        s: _you_ratio(" ".join(t["text"] for t in turns if t["speaker"] == s))
        for s in raw_speakers
    }
    doctor = max(raw_speakers, key=lambda s: ratios[s])

    for turn in turns:
        turn["speaker"] = "DOCTOR" if turn["speaker"] == doctor else "PATIENT"

    return turns