# Phase 4 — Output Layer

- [x] Implement `/speak` endpoint: fused label + mismatch flag in → short spoken phrase out (e.g. "calm, but sounds forced" if mismatch flag true, otherwise the 1-2 word label)
- [x] Wire gTTS as primary TTS engine (generates natural MP3 stream)
- [x] Wire pyttsx3 as offline fallback, triggered when no internet connection is detected (generates SAPI5 WAV)
- [x] Test both online and simulated-offline conditions


## Deliverable
Working TTS endpoint tested under both online and offline conditions.

## Deviations
(log any deviation from the prescribed stack here, with a one-line reason)
