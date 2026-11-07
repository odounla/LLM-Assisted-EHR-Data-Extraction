import whisper

model = whisper.load_model('large-v3')


def transcribe(audio_path: str) -> dict:
    """Transcribe one .mp3 with Whisper large-v3.

    Returns the full Whisper result dict:
      - 'text': the whole transcript as one string
      - 'segments': list of chunks, each with 'start', 'end', 'text'
        and a 'words' list (each word has 'start', 'end', 'word');
        align.py uses the word timestamps to match each word to a speaker

    Settings:
      - word_timestamps=True: a start/end time for every word, so speaker
        changes inside a chunk are handled correctly
      - condition_on_previous_text=False: stops Whisper from feeding its
        own earlier output back in, which causes repetition loops
    """
    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    return result


if __name__ == "__main__":
    test_file = "data/corpus/Abel832_Lubowitz58_63f00385-5bdc-66fb-5b6d-174facc56175.mp3"
    result = transcribe(test_file)
    print(result["text"][:300])
    print("Number of segments:", len(result["segments"]))
    print("First segment words:", result["segments"][0]["words"][:3])