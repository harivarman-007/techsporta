# Phase 1 — Individual Modality Pipelines

- [x] Implement `/face-emotion` endpoint: webcam frame in → MediaPipe Face Mesh crop → `trpakov/vit-face-expression` (FP16, CUDA) → emotion label + confidence JSON
- [x] Implement `/speech-emotion` endpoint: audio clip in → VAD trims silence (webrtcvad) → `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` (FP16, CUDA) → emotion label + confidence JSON
- [x] Implement `/transcript` endpoint: audio clip in → faster-whisper (base model, CPU int8) → transcript text JSON
- [x] Write standalone test scripts for each endpoint using sample image/audio files and live stream
- [x] Log VRAM usage after each model loads (`nvidia-smi`), confirm combined total stays under ~2GB

## Deliverable
Three working, independently testable endpoints, verified with real-time video/audio stream and live 30 FPS face mapping HUD. VRAM usage verified at ~2.1 GB on RTX 3050.

## Deviations
- `faster-whisper`: Run on CPU with `int8` quantization (takes <150ms) to avoid missing `cublas64_12.dll` errors on Windows with PyTorch CUDA 11.8, and to keep GPU VRAM reserved for ViT and Wav2Vec2.
- `MediaPipe 1.0.1`: Uses `FaceLandmarker` with `face_landmarker.task` for 478 3D landmarks + 52 FACS blendshapes.
- `FACS + Neutral Baseline Calibration`: Integrated Claude's `NeutralCalibrator`, multi-cue AU heuristics, and square crop `crop_face_from_landmarks` to eliminate resting-face angry/sad bias.
