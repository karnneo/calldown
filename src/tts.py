"""Spoken acknowledgements via Piper (local neural TTS).

Piper is a self-contained, fully-offline TTS engine that ships its own
espeak-ng phonemizer data and runs the same way on Windows/Linux/Mac,
unlike OS-level engines (e.g. Windows SAPI5) whose available voices depend
on what's registered on that particular machine. The chosen voice model
(tts_voice in settings.yaml, any name from
https://huggingface.co/rhasspy/piper-voices) is downloaded once into
voice_dir and reused after that - no internet needed at runtime.

Synthesis + playback happens on a dedicated background thread so a long
acknowledgement never delays listening for the next push-to-talk press.
"""
import os
import queue
import threading
import traceback
import random
from pathlib import Path

import numpy as np
import sounddevice as sd
from piper.config import SynthesisConfig
from piper.download_voices import download_voice
from piper.voice import PiperVoice


class Speaker:
    def __init__(
        self,
        enabled: bool,
        voice_name: str,
        voice_dir: str,
        speed: float = 1.0,
        volume: float = 1.0,
    ):
        self.enabled = enabled
        self._voice_name = voice_name
        self._voice_dir = voice_dir
        self._speed = speed
        self._volume = volume
        self._queue: "queue.Queue[str]" = queue.Queue()
        if self.enabled:
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            os.makedirs(self._voice_dir, exist_ok=True)
            model_path = os.path.join(self._voice_dir, f"{self._voice_name}.onnx")
            if not os.path.exists(model_path):
                print(f"[tts] downloading voice '{self._voice_name}' (one-time)...")
                download_voice(self._voice_name, Path(self._voice_dir))
                print("[tts] voice downloaded")

            voice = PiperVoice.load(model_path)
            syn_config = SynthesisConfig(length_scale=self._speed, volume=self._volume)
        except Exception:
            print("[tts] failed to initialize speech engine:")
            traceback.print_exc()
            return

        while True:
            text = self._queue.get()
            try:
                for chunk in voice.synthesize(text, syn_config=syn_config):
                    audio = chunk.audio_int16_array
                    if chunk.sample_channels == 1:
                        # Some multichannel surround devices (e.g. gaming
                        # headset DACs presenting 5.1/6-channel virtual
                        # surround outputs) silently drop raw mono PCM
                        # written over the legacy MME/WinMM API. Piper's
                        # voices are mono, so always upmix to stereo before
                        # playback rather than relying on device-specific
                        # channel handling.
                        audio = np.column_stack([audio, audio])
                    sd.play(audio, chunk.sample_rate)
                    sd.wait()
            except Exception:
                print("[tts] failed to speak:")
                traceback.print_exc()

    def speak(self, text: str):
        if not self.enabled or not text:
            return
        self._queue.put(text)


def resolve_ack(command: dict, default_ack: str) -> str:
    """Pick the acknowledgement phrase for a matched command.

    A profile entry may set "ack" to a single string, or to a list of
    strings to vary the reply (one is picked at random each time). Falls
    back to default_ack (settings.yaml) if the command sets neither.
    """
    ack = command.get("ack")
    if isinstance(ack, list):
        return random.choice(ack) if ack else default_ack
    if isinstance(ack, str) and ack:
        return ack
    return default_ack
