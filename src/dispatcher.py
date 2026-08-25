"""Translates a stratagem code (e.g. "DDUR") into actual keypresses.

Holds the menu key (default: Left Ctrl), taps each directional key in
sequence with a small delay so the game registers each input distinctly,
then releases the menu key. Timing is deliberately configurable and slightly
randomized — games can drop inputs that arrive faster than a human could
physically type them.
"""
import random
import time

import pydirectinput

pydirectinput.PAUSE = 0  # we control timing manually


class Dispatcher:
    def __init__(
        self,
        menu_key: str,
        direction_keys: dict,
        key_press_duration_ms: int,
        key_gap_ms: int,
        menu_settle_delay_ms: int,
        menu_release_delay_ms: int,
        jitter_ms: int = 0,
    ):
        self.menu_key = menu_key
        self.direction_keys = direction_keys  # e.g. {"U": "w", "D": "s", "L": "a", "R": "d"}
        self.press_duration = key_press_duration_ms / 1000.0
        self.key_gap = key_gap_ms / 1000.0
        self.menu_settle_delay = menu_settle_delay_ms / 1000.0
        self.release_delay = menu_release_delay_ms / 1000.0
        self.jitter = jitter_ms / 1000.0

    def _wait(self, base_seconds: float):
        if self.jitter <= 0:
            time.sleep(base_seconds)
            return
        # Randomize +/- jitter around the base delay, floored at 0.
        offset = random.uniform(-self.jitter, self.jitter)
        time.sleep(max(0.0, base_seconds + offset))

    def execute(self, code: str):
        print(f"[dispatch] entering code: {code}")
        pydirectinput.keyDown(self.menu_key)
        try:
            self._wait(self.menu_settle_delay)  # let the game register the menu opening
            for direction in code:
                key = self.direction_keys[direction]
                pydirectinput.keyDown(key)
                self._wait(self.press_duration)
                pydirectinput.keyUp(key)
                self._wait(self.key_gap)
            self._wait(self.release_delay)
        finally:
            pydirectinput.keyUp(self.menu_key)

    def press_key(self, key: str):
        """Tap a single keybind directly (no stratagem menu) — e.g. Stim."""
        print(f"[dispatch] pressing key: {key}")
        pydirectinput.keyDown(key)
        self._wait(self.press_duration)
        pydirectinput.keyUp(key)
