"""Calldown: semantic voice-command caller for games.

Pipeline: push-to-talk record -> Whisper STT -> semantic intent match -> keypress dispatch.
Which game's commands are active is controlled by `profile` in settings.yaml,
pointing at a file under profiles/.
"""
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from audio import record_while_held
from dispatcher import Dispatcher
from matcher import IntentMatcher
from stt import Transcriber
from tts import Speaker, resolve_ack

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")

BANNER = r"""
   _____      _ _     _
  / ____|    | | |   | |
 | |     __ _| | | __| | _____      ___ __
 | |    / _` | | |/ _` |/ _ \ \ /\ / / '_ \
 | |___| (_| | | | (_| | (_) \ V  V /| | | |
  \_____\__,_|_|_|\__,_|\___/ \_/\_/ |_| |_|
           Drop the Pain !!!!!!
"""


def load_settings():
    with open(os.path.join(ROOT_DIR, "settings.yaml"), "r") as f:
        return yaml.safe_load(f)


def main():
    print(BANNER)
    cfg = load_settings()
    profile_path = os.path.join(ROOT_DIR, "profiles", f"{cfg['profile']}.json")

    print(f"[init] profile: {cfg['profile']}")
    print("[init] loading Whisper model...")
    transcriber = Transcriber(
        cfg["whisper_model"], cfg["whisper_device"], cfg["whisper_compute_type"]
    )

    print("[init] loading embedding model + command phrasebook...")
    matcher = IntentMatcher(profile_path, cfg["embedding_model"])

    dispatcher = Dispatcher(
        cfg["menu_key"],
        cfg["direction_keys"],
        cfg["key_press_duration_ms"],
        cfg["key_gap_ms"],
        cfg["menu_settle_delay_ms"],
        cfg["menu_release_delay_ms"],
        cfg.get("jitter_ms", 0),
    )

    voice_dir = os.path.join(ROOT_DIR, "models", "piper")
    speaker = Speaker(
        cfg.get("tts_enabled", True),
        cfg.get("tts_voice", "en_US-lessac-medium"),
        voice_dir,
        cfg.get("tts_speed", 1.0),
        cfg.get("tts_volume", 1.0),
    )
    default_ack = cfg.get("default_ack", "")

    log_path = os.path.join(ROOT_DIR, cfg["log_unmatched_to"])

    print(f"[ready] hold '{cfg['ptt_key']}' and say a command. Ctrl+C to quit.")
    while True:
        try:
            audio = record_while_held(cfg["ptt_key"], cfg.get("ptt_device", "keyboard"))
            text = transcriber.transcribe(audio)
            if not text:
                continue

            result = matcher.match(
                text, cfg["confidence_threshold"], cfg["ambiguity_margin"]
            )

            if result.command is None:
                print(
                    f"[reject] no confident match (best={result.score:.2f}, "
                    f"runner_up={result.runner_up_score:.2f}) for: '{text}'"
                )
                with open(log_path, "a") as f:
                    f.write(f"{text}\t best={result.score:.2f}\n")
                continue

            print(
                f"[match] '{text}' -> {result.command['name']} "
                f"(score={result.score:.2f})"
            )
            speaker.speak(resolve_ack(result.command, default_ack))
            if "code" in result.command:
                dispatcher.execute(result.command["code"])
            else:
                dispatcher.press_key(result.command["key"])

        except KeyboardInterrupt:
            print("\n[exit] shutting down")
            break


if __name__ == "__main__":
    main()
