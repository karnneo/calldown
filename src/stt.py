"""Speech-to-text via faster-whisper."""
import numpy as np
from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, model_size: str, device: str, compute_type: str):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        segments, _info = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,          # greedy decoding: fastest, fine for short commands
            vad_filter=True,      # trims leading/trailing silence
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        print(f"[stt] '{text}'")
        return text
