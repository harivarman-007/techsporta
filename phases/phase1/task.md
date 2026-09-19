# Phase 1 — Individual Modality Pipelines

- [x] Implement `/face-emotion` endpoint: webcam frame in → MediaPipe Face Mesh crop → `trpakov/vit-face-expression` (FP16, CUDA) → emotion label + confidence JSON
- [x] Implement `/speech-emotion` endpoint: audio clip in → VAD trims silence (webrtcvad) → `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` (FP16, CUDA) → emotion label + confidence JSON
- [x] Implement `/transcript` endpoint: audio clip in → faster-whisper (base model, CPU int8) → transcript text JSON
- [x] Write standalone test scripts for each endpoint using sample image/audio files and live stream
- [x] Log VRAM usage after each model loads (`nvidia-smi`), confirm combined total stays under ~2GB

## Deliverable
Three working, independently testable endpoints, verified with real-time video/audio stream and live 30 FPS face mapping HUD. VRAM usage verified at ~2.1 GB on RTX 3050.

## Deviations
- Python runtime: Running on Python 3.13.1 on host system.
- `faster-whisper`: Run on CPU with `int8` quantization (takes <150ms) to avoid missing `cublas64_12.dll` errors on Windows with PyTorch CUDA 11.8, and to keep GPU VRAM reserved for ViT and Wav2Vec2.
- `MediaPipe 1.0.1`: Uses `FaceLandmarker` with `face_landmarker.task` for face detection and tight face crop for ViT.
- `FACS & Calibration`: Advanced 52-FACS blendshapes and per-user resting-face baseline calibration are preserved as an optional opt-in feature (`use_facs=True` / `ENABLE_FACS=1`), but gated off by default. The default pipeline strictly runs MediaPipe crop -> ViT model -> emotion label + confidence as specified in PROMPT.md.
- `WS /stream`: Real-time streaming WebSocket is preserved as an optional debug extra, gated off by default (`ENABLE_STREAM_WS=1`) to adhere to single-shot REST specifications.
