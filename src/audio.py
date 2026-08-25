"""Push-to-talk audio capture.

Records mic audio only while the configured push-to-talk control is held
down, so it plays nicely with Discord/game voice chat and doesn't pick up
open-mic noise mid-firefight. Supports either a keyboard key or a native
mouse button (Mouse 4/5 etc.) as the trigger.
"""
import keyboard
import mouse
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000


def _is_pressed(ptt_device: str, ptt_key: str) -> bool:
    if ptt_device == "mouse":
        return mouse.is_pressed(ptt_key)
    return keyboard.is_pressed(ptt_key)


def _wait_for_press(ptt_device: str, ptt_key: str, poll_seconds: float = 0.01):
    if ptt_device == "keyboard":
        keyboard.wait(ptt_key)
        return
    # mouse library has no blocking wait(), so poll.
    while not mouse.is_pressed(ptt_key):
        sd.sleep(int(poll_seconds * 1000))


def record_while_held(
    ptt_key: str, ptt_device: str = "keyboard", max_seconds: float = 8.0
) -> np.ndarray:
    """Blocks until the PTT control is pressed, records until it's released.

    ptt_device: "keyboard" or "mouse".
    ptt_key: a keyboard key name (e.g. "f13") when ptt_device is "keyboard",
             or a mouse button name when ptt_device is "mouse" — one of
             "left", "right", "middle", "x" (Mouse 4 / back), "x2" (Mouse 5 / forward).

    Returns mono float32 audio at 16kHz, ready for Whisper.
    """
    print(f"[audio] waiting for {ptt_device} '{ptt_key}' ...")
    _wait_for_press(ptt_device, ptt_key)

    frames = []

    def callback(indata, _frames, _time, _status):
        frames.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback
    ):
        print("[audio] recording...")
        # Keep recording while the control is physically held, up to max_seconds.
        elapsed = 0.0
        step = 0.02
        while _is_pressed(ptt_device, ptt_key) and elapsed < max_seconds:
            sd.sleep(int(step * 1000))
            elapsed += step

    if not frames:
        return np.zeros(0, dtype=np.float32)

    audio = np.concatenate(frames, axis=0).flatten()
    print(f"[audio] captured {len(audio) / SAMPLE_RATE:.2f}s")
    return audio
