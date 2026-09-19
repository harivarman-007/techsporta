# EmpathAI — Project Plan

Multimodal emotion recognition assistant for visually impaired users. Face + speech + spoken-context signals are fused into a short spoken emotional summary, with explicit detection of mismatches between channels (e.g. calm words spoken in a tense tone).

## Phase checklist

- [x] **Phase 0 — Project scaffolding**
  Repo structure, backend/frontend skeletons running, env config in place.

- [x] **Phase 1 — Individual modality pipelines**
  Face emotion (ViT + MediaPipe FACS), speech emotion (Wav2Vec2-XLSR), transcript (faster-whisper) — each working and tested standalone and in live stream.

- [x] **Phase 2 — Context reasoning layer**
  Gemini Flash structured-output call turning transcript text into a sentiment/context read.


- [x] **Phase 3 — Fusion layer**
  Trained MLP fusion head + Gemini-based fusion prompt path, both producing a final label + mismatch flag.


- [x] **Phase 4 — Output layer**
  gTTS/pyttsx3 TTS turning fused result into a short spoken phrase.


- [x] **Phase 5 — End-to-end pipeline wiring**
  Single `/analyze` endpoint chaining all of the above, with per-stage latency logging.


- [x] **Phase 6 — Frontend integration**
  Minimal React + Tailwind UI: record button, status, transcript/label display, spoken output.


- [x] **Phase 7 — Testing, accuracy validation, and demo prep**
  Congruent/mismatched test cases, real-condition testing, latency target, backup demo scenarios, README.

## Notes
- Hardware target: RTX 3050, 4GB VRAM — all models run FP16, combined usage must stay under ~2GB.
- No automated browser screenshots or headless browser visual checks at any point.
- Each phase's detailed subtasks live in `/phases/phaseN/task.md`.
