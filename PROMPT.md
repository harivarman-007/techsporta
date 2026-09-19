# Project: EmpathAI — Multimodal Emotion Recognition for Visually Impaired Users

## IMPORTANT — Agent behavior constraints
- Do NOT use automated browser screenshots or visual browser verification at any point in this build.
- Do NOT launch a headless browser to check frontend rendering.
- Verify frontend work by reading the rendered HTML/component output and console logs only, never by screenshotting.
- Work strictly phase-by-phase. Do not start Phase N+1 until Phase N's plan.md items are all checked off.
- Use the `plan.md` at project root (overall roadmap) and the per-phase `task.md` files in `/phases/phaseN/task.md`, updated as work progresses — do not skip this documentation step.

## Project overview
Build a multimodal emotion recognition system that helps visually impaired users perceive the emotional state of a person they're talking to. The system captures a short video+audio clip of a conversation partner, analyzes facial expression, vocal tone, and spoken content, fuses all three signals, and speaks back a short (1-3 word) emotional summary via TTS — including flagging when tone/words/expression disagree (e.g. sarcasm, forced calm).

## Tech stack (use exactly these unless a listed package fails to install)
- **Backend**: Python 3.11, FastAPI, uvicorn
- **Face detection**: MediaPipe Face Mesh
- **Facial emotion model**: Hugging Face `trpakov/vit-face-expression` (ViT-base), loaded in FP16 on CUDA
- **Speech emotion model**: Hugging Face `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`, loaded in FP16 on CUDA
- **Voice activity detection**: `torchaudio` VAD or `webrtcvad`
- **Speech-to-text**: `faster-whisper` (base model, CTranslate2 backend)
- **Context reasoning**: Google Gemini API (`gemini-2.5-flash`), using function calling / structured JSON output
- **Fusion**: small PyTorch MLP (3 layers, softmax output over emotion classes + a binary "mismatch" flag), plus a Gemini-based cross-check fusion path for comparison
- **TTS**: gTTS primary, pyttsx3 offline fallback
- **Frontend**: React + Tailwind CSS, single page, minimal UI (record button, live status text, spoken-output transcript display)
- **Hardware target**: NVIDIA RTX 3050 (4GB VRAM) — all models must run in FP16, verify combined VRAM usage stays under ~2GB with `nvidia-smi` logging

## Phase structure

### Phase 0 — Project scaffolding
- Set up repo structure: `/backend`, `/frontend`, `/models`, `/phases`
- Confirm root `plan.md` matches the phases below with checkboxes
- Set up Python venv, install backend dependencies
- Set up React app with Tailwind, no functionality yet
- Create `.env.example` for Gemini API key
- Deliverable: empty but running FastAPI server + empty but running React dev server, confirmed via curl/logs, not browser screenshot

### Phase 1 — Individual modality pipelines (build and test in isolation)
- Implement face capture endpoint: webcam frame in → MediaPipe crop → ViT model → emotion label + confidence, returned as JSON
- Implement speech capture endpoint: audio clip in → VAD trim silence → Wav2Vec2-XLSR model → emotion label + confidence, returned as JSON
- Implement transcript endpoint: same audio clip → faster-whisper → transcript text, returned as JSON
- Write a standalone test script for each of the three endpoints using sample files (no live capture yet)
- Log VRAM usage after each model loads via `nvidia-smi`, confirm total stays under 2GB
- Deliverable: three working, independently testable endpoints, each verified with sample inputs and logged output — no fusion yet

### Phase 2 — Context reasoning layer
- Implement Gemini Flash call using function calling / structured output: input = transcript text, output = JSON `{sentiment: str, confidence: float}`
- Test with several sample transcripts covering calm, urgent, and sarcastic-sounding text
- Deliverable: working context endpoint returning structured JSON, tested against multiple sample inputs

### Phase 3 — Fusion layer
- Build a small PyTorch MLP fusion head: input = [face_label_vector, speech_label_vector, context_sentiment_vector] concatenated, output = final emotion label + mismatch flag (boolean)
- Since no large labeled fusion dataset exists, generate a small synthetic labeled set (manually authored combinations of face/speech/context labels mapped to expected fused output + mismatch flag) for initial training — document this limitation clearly in `task.md`
- In parallel, implement a Gemini-based fusion prompt path: send all three modality outputs to Gemini Flash, ask it to reconcile them into a final label + mismatch flag, using structured output
- Both fusion paths should be callable independently so they can be compared during testing
- Deliverable: two working fusion approaches (trained MLP + LLM reasoning), both tested against the same set of sample modality combinations, results logged for comparison

### Phase 4 — Output layer
- Implement TTS endpoint: fused label + mismatch flag in → short spoken phrase out (e.g. "calm, but sounds forced" if mismatch flag is true, otherwise just the 1-2 word label)
- gTTS primary, pyttsx3 fallback if no internet connection detected
- Deliverable: working TTS endpoint tested with both online and offline (simulated no-internet) conditions

### Phase 5 — End-to-end pipeline wiring
- Chain Phase 1-4 endpoints into a single `/analyze` endpoint: accepts a short video+audio clip, runs all modality models, fusion, and TTS, returns final spoken output + all intermediate labels for debugging
- Add clear latency logging at each pipeline stage (face inference time, speech inference time, transcript time, context call time, fusion time, TTS time) so bottlenecks are visible
- Deliverable: single working end-to-end endpoint tested with several sample clips, full latency breakdown logged

### Phase 6 — Frontend integration
- Build minimal React frontend: record button (camera + mic), status indicator during processing, display of transcript + detected face/speech/context labels + final spoken output as text
- Wire frontend to call the `/analyze` endpoint
- Verify rendering correctness by inspecting component output/console logs only — no browser screenshot tooling
- Ensure the app is reachable over local Wi-Fi from a phone browser (bind FastAPI to `0.0.0.0`, note the local IP in README)
- Deliverable: working frontend, tested by manual interaction, confirmed via logs/network tab output described in text, not screenshots

### Phase 7 — Testing, accuracy validation, and demo prep
- Record test cases: congruent emotion (happy face + happy tone + happy words) and mismatched cases (calm words, tense tone) — verify mismatch flag triggers correctly
- Test under realistic conditions: laptop mic, ambient noise, normal lighting
- Confirm total end-to-end latency is acceptable for a live demo (target: under 5-8 seconds per analysis)
- Record a backup demo video of a successful run
- Write final `README.md` summarizing architecture, tech stack, and how to run the project, referencing the literature-survey limitations this project addresses (high compute cost, limited contextual understanding, limited emotion reasoning — cite generically without inventing sources)

## General rules for every phase
- Update the relevant `task.md` checkboxes as each subtask completes — do not batch updates at the end of a phase
- Do not proceed to the next phase until the current phase's deliverable is confirmed working
- Prefer clear, well-commented code over cleverness — this needs to be explainable in a judging Q&A
- Flag any deviation from the tech stack above in `task.md` with a one-line reason
- No automated browser screenshots or headless browser visual checks at any point, in any phase
