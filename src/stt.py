"""Speech-to-text via faster-whisper."""
import importlib.util
import os


def _add_nvidia_dll_dirs():
    """pip's nvidia-cublas-cu12 / nvidia-cudnn-cu12 / nvidia-cuda-nvrtc-cu12
    packages ship the CUDA DLLs ctranslate2 needs, but don't add them to
    PATH. Without this, loading the Whisper model on CUDA fails with e.g.
    "cublas64_12.dll is not found or cannot be loaded" — not at model load
    (ctranslate2 loads those lazily), but the first time inference actually
    runs on the GPU.

    os.add_dll_directory() alone does NOT fix this: it only affects Python's
    own extension-module (.pyd) loader, not the plain LoadLibrary calls
    ctranslate2's CUDA backend makes internally. Those follow the classic
    Windows DLL search order, which does check PATH — so that's what needs
    updating.
    """
    if os.name != "nt":
        return
    bin_dirs = []
    for pkg in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc"):
        spec = importlib.util.find_spec(pkg)
        if spec and spec.submodule_search_locations:
            bin_dir = os.path.join(spec.submodule_search_locations[0], "bin")
            if os.path.isdir(bin_dir):
                bin_dirs.append(bin_dir)
    if bin_dirs:
        os.environ["PATH"] = os.pathsep.join(bin_dirs) + os.pathsep + os.environ["PATH"]


_add_nvidia_dll_dirs()

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
