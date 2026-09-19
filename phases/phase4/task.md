# Phase 4 — Output Layer

- [x] Implement `/speak` endpoint: fused label + mismatch flag in → short spoken phrase out (e.g. "calm, but sounds forced" if mismatch flag true, otherwise the 1-2 word label)
- [x] Wire gTTS as primary TTS engine (generates natural MP3 stream)
- [x] Wire pyttsx3 as offline fallback, triggered when no internet connection is detected (generates SAPI5 WAV)
- [x] Test both online and simulated-offline conditions


## Deliverable
Working TTS endpoint tested under both online and offline conditions.

## Deviations
- Python runtime: Running on Python 3.13.1 on host system.
- Spoken Summary: Output strictly restricted to 1–3 words (e.g. "happy", "calm, but sounds forced") by default per PROMPT.md; multi-sentence descriptive narration moved behind optional flag `allow_long_narration=False`.
