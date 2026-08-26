<img width="627" height="627" alt="calldown-logo" src="https://github.com/user-attachments/assets/bccd1b3a-f890-4f41-a2e1-7f9c464e0060" />


# Calldown

Semantic voice-command caller for games — say things naturally and it maps
them to in-game actions, instead of requiring one exact trigger phrase per
command like VoiceAttack does.

Started as a Helldivers 2 stratagem caller, structured so any game can be
added as its own profile.

This is really early in development, not for public use. 
I've hacked this together with a bit of python code and the hallucinations 
of an AI.  Use at your own risk
It's been developed by an Aussie so check the JSON file and 
remove colourful language if your sensitive.

If it's proves useful I'll develop it more and build it as a neat package an 
maybe add a Linux version once Windows shits me too much an i move over to 
gaming on linux.

## How it works

1. Hold the push-to-talk key (`ptt_key` in `settings.yaml`) and speak.
2. `faster-whisper` transcribes the utterance locally (no cloud, no API key).
3. The transcript is embedded with a sentence-transformer model and compared
   against every command's example phrases by cosine similarity.
4. If there's a confident, unambiguous match, the software speaks a short
   acknowledgement and the corresponding key sequence is dispatched to the
   game.

Everything runs locally — no internet connection needed at runtime.

## Folder layout

```
calldown/
  settings.yaml          <- your machine/hardware settings + active profile
  profiles/
    helldivers2.json      <- Helldivers 2 stratagem definitions
    (add more games here, e.g. elite-dangerous.json)
  src/
    main.py               <- entry point
    audio.py, stt.py, matcher.py, dispatcher.py
  requirements.txt
```

## Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

You'll need a CUDA-capable GPU for real-time Whisper performance (set
`whisper_device: cpu` in settings.yaml if you don't have one, but expect
slower responses). First run downloads the Whisper and embedding models.

**Run as Administrator on Windows** — input injection needs elevated
permissions in most games.

```
python src\main.py
```

## Before your first mission

1. In `settings.yaml`, set `ptt_key` to your push-to-talk key/button.
2. Confirm `direction_keys` in `settings.yaml` matches your in-game control
   scheme (default Helldivers 2 keybind is WASD while holding Left Ctrl).
3. **Verify the codes in `profiles/helldivers2.json` against what the game
   shows on your loadout screen.** Seeded with commonly-cited codes, but
   double-check before relying on it mid-mission — patches can shift things.

## Adding commands to a profile

Each entry in a profile's `commands` list looks like:

```json
{
  "name": "Orbital Gas Strike",
  "code": "DRDL",
  "aliases": ["gas strike", "orbital gas", "gas them out", "throw gas"]
}
```

- `code`: sequence of `U`/`D`/`L`/`R` (or whatever your dispatcher maps —
  see `direction_keys` in settings.yaml) matching the in-game display order.
- `aliases`: as many natural phrasings as you can think of. More variety
  here = better semantic recognition — you don't need exact matches.

For an action with its own dedicated keybind (not entered through the
stratagem menu, e.g. Stim) use `key` instead of `code`:

```json
{
  "name": "Stim",
  "key": "v",
  "aliases": ["stim", "heal", "use stim", "heal up"]
}
```

- `key`: the single key to tap directly, matching your in-game keybind.

## Spoken acknowledgements

When a command is confidently matched, Calldown speaks it back to you so
you know it heard you correctly without having to look at the console.

This uses [Piper](https://github.com/OHF-Voice/piper1-gpl), a small
self-contained neural TTS engine — not the operating system's own voices
— so it sounds and behaves the same on Windows, Linux, and Mac, and
doesn't depend on any OS-level voice setup (unlike Windows SAPI5, which
sometimes has no usable voices registered out of the box). The voice
model set in `tts_voice` is downloaded once into `models/piper/` the
first time you run Calldown (needs internet for that one download; fully
offline after that).

Add an `"ack"` field to any command in a profile to customize the reply:

```json
{
  "name": "Stim",
  "key": "v",
  "aliases": ["stim", "heal", "use stim"],
  "ack": ["Stimming", "Stim deployed"]
}
```

- A single string always says the same thing.
- A list of strings picks one at random each time, for variety.
- If a command has no `ack`, `default_ack` in `settings.yaml` is spoken
  instead.

Tune the voice or turn acknowledgements off entirely in `settings.yaml`:

- `tts_enabled` — set to `false` to disable spoken acknowledgements.
- `tts_voice` — any voice name from
  [the Piper voices list](https://huggingface.co/rhasspy/piper-voices)
  (e.g. `"en_US-lessac-medium"`, `"en_GB-alan-medium"`).
- `tts_speed` — `<1.0` = faster speech, `>1.0` = slower.
- `tts_volume` — multiplier on the spoken audio, e.g. `1.5` = louder.
- `default_ack` — fallback phrase for commands without their own `ack`.

## Adding a new game

1. Create `profiles/<game-name>.json` with a `commands` list in the same
   shape as above (the `code` values just need to mean something to your
   `dispatcher.py` — for games without directional-code menus you may want
   a simpler dispatcher that just sends a single keybind per command).
2. Set `profile: "<game-name>"` in `settings.yaml`.

## Tuning recognition

In `settings.yaml`:

- `confidence_threshold` — raise if it's firing on things you didn't mean;
  lower if it's rejecting commands it should catch.
- `ambiguity_margin` — if two commands sound similar and it keeps guessing
  wrong between them, raise this so it rejects instead of guessing.
- Check `unmatched_utterances.log` after a session — anything rejected is
  logged there so you can add it as a new alias for the command you meant.

## Tuning keypress timing

If the game is dropping or misreading inputs, it's almost always because the
keypresses are arriving faster than a real keypress would. Adjust these in
`settings.yaml`:

- `key_press_duration_ms` — how long each directional key is physically held
  down. Increase if inputs seem to get skipped.
- `key_gap_ms` — pause between releasing one direction key and pressing the
  next. Increase if consecutive same-direction inputs merge into one.
- `menu_settle_delay_ms` — pause after opening the stratagem menu before the
  first direction is sent. Increase if the very first input in a sequence
  is the one that keeps getting missed.
- `menu_release_delay_ms` — pause after the last direction before releasing
  the menu key. Increase if the last input in a sequence gets cut off.
- `jitter_ms` — random +/- variance added to every delay above, so the
  cadence doesn't look perfectly robotic. Set to `0` to disable.

If you're still seeing misfires after raising these, bump them up
significantly (e.g. double everything) to confirm reliability first, then
dial back down for speed once you know the ceiling.

## Notes on latency

Full loop (record → transcribe → match → dispatch) should land well under a
second on a mid-range GPU with the `small` Whisper model. If it feels
sluggish:
- Drop to `whisper_model: "tiny"` (less accurate, faster)
- Make sure `whisper_device: "cuda"` is actually picking up your GPU
- Keep push-to-talk recordings short — say the command, release quickly
