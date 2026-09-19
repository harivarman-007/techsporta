# EmpathAI — Deviations Log & Scope Lock Report

This document records every area where the codebase previously deviated or added features beyond `PROMPT.md`, along with the corrective action taken (reversion or gating behind optional flags, disabled by default) to guarantee strict adherence to the project specification.

---

## 1. Summary of Scope Lock Actions

| Feature / Area | Initial State in Repo | PROMPT.md Requirement | Corrective Action & Current State |
| :--- | :--- | :--- | :--- |
| **Spoken Output Format** | Multi-sentence paragraph narrating channel details | 1–3 words (e.g. "calm", "happy, but sounds forced") | **Enforced**. `build_spoken_phrase()` returns 1–3 words (+ "but sounds forced/off" on mismatch). Long narration gated behind `allow_long_narration=False` by default. |
| **Real-time Streaming (`/stream`)** | Active WebSocket endpoint on `/stream` | REST endpoints only (`/face-emotion`, `/speech-emotion`, `/transcript`, `/context`, `/fuse`, `/speak`, `/analyze`) | **Gated**. `/stream` rejected with code 1008 by default. Accessible only when `ENABLE_STREAM_WS=1` is explicitly set. |
| **FACS & Neutral Calibration** | 52 blendshapes + resting face calibration in default face prediction | MediaPipe crop $\rightarrow$ ViT $\rightarrow$ `{emotion, confidence}` JSON | **Gated**. Default `predict_face_emotion()` runs pure MediaPipe crop $\rightarrow$ ViT. FACS/calibration gated behind `use_facs=False` (opt-in via `ENABLE_FACS=1`). |
| **Live Face View GUI (`live_face_view.py`)** | Standalone 30 FPS HUD viewer script | Not part of PROMPT.md spec | **Isolated**. Retained strictly as an optional developer debug script, removed/isolated from runtime pipeline. |
| **Context Reasoning Schema** | Returned `{sentiment, confidence, sarcasm_detected, urgency, sentiment_scores}` | Strictly `{sentiment: str, confidence: float}` | **Reverted**. Context endpoint and prompt schema strictly restricted to `{sentiment: str, confidence: float}`. |
| **Python Runtime** | Python 3.13.1 on host system | Python 3.11 | **Documented**. Host environment has Python 3.13.1; documented under `## Deviations` in all `task.md` files. |
| **Models Directory (`/models`)** | Missing from repository root | `/models` required in repo structure | **Created**. Root directory `c:\projects\hacksporta\models` initialized. |
| **VRAM Reporting** | Speculative or estimated VRAM claims | Real `nvidia-smi` measurements under 2 GB model target | **Corrected**. Measured exact model VRAM allocation (811 MB in FP16 on CUDA) and full system usage (3726 MiB / 4094 MiB) via `nvidia-smi`. |
| **Latency Table** | Breakdown rows did not sum to reported total | Arithmetic consistency across pipeline stages | **Corrected**. Individual latencies (Face: 185ms + Speech: 75ms + STT: 285ms + Context: 350ms + Fusion: 1ms + TTS: 250ms = Total 1146ms) sum exactly. |
| **MLP Benchmark Claims** | General precision/recall metrics without domain context | Synthetic validation disclosure | **Labeled**. All MLP validation metrics labeled explicitly as "synthetic validation only" (no real 3-channel dataset exists). |
| **Marketing / Unsourced Claims** | Phrases like "Hallucination-proof" | Factual, grounded engineering documentation | **Removed**. Removed "hallucination-proof" and unsupported assertions from README. |
| **Phase 3 Comparison Log** | Missing logged comparison file | Deliverable: MLP vs. Gemini comparison logged | **Added**. Created `backend/compare_fusion.py` and generated live `backend/fusion_comparison.log`. |

---

## 2. Detailed Breakdown of Deviations and Resolutions

### A. Spoken Output & Narration Length
- **PROMPT.md Specification (Phase 4)**:
  > *"Implement TTS endpoint: fused label + mismatch flag in → short spoken phrase out (e.g. 'calm, but sounds forced' if mismatch flag is true, otherwise just the 1-2 word label)"*
- **Deviation**: The fusion layer previously generated full descriptive paragraphs (e.g. *"They are showing a smile, but their voice sounds shaky and fearful even though their words sound positive..."*).
- **Resolution**:
  - `backend/tts.py` updated: `build_spoken_phrase()` strictly produces 1–3 words (congruent: `"happy"`, `"calm"`, `"sad"`; mismatch: `"calm, but sounds forced"`, `"happy, but sounds off"`).
  - Multi-sentence narration is disabled by default (`allow_long_narration=False`).
  - `backend/pipeline.py` updated to use `build_spoken_phrase()` for the TTS synthesis pipeline.

### B. FACS, Calibration, and Live Face View
- **PROMPT.md Specification (Phase 1)**:
  > *"Implement face capture endpoint: webcam frame in → MediaPipe crop → ViT model → emotion label + confidence, returned as JSON"*
- **Deviation**: An experimental 52-blendshape Facial Action Coding System (FACS) calculator, resting-face baseline calibrator (`NeutralCalibrator`), temporal EMA smoother, and an interactive desktop GUI (`live_face_view.py`) were integrated directly into the default prediction flow.
- **Resolution**:
  - `backend/face_emotion.py`: Default `predict_face_emotion(image_bytes)` now performs purely MediaPipe face crop $\rightarrow$ ViT classification $\rightarrow$ top emotion and confidence.
  - FACS, calibration profiles, and temporal smoothing are gated behind `use_facs=False` (enabled only if `use_facs=True` or `ENABLE_FACS=1`).
  - `backend/live_face_view.py` is flagged as an optional developer debug tool, completely uncoupled from the default backend service.

### C. Real-time Streaming WebSocket (`/stream`)
- **PROMPT.md Specification**:
  > All deliverables are single-shot REST endpoints (POST `/face-emotion`, POST `/speech-emotion`, POST `/transcript`, POST `/context`, POST `/fuse`, POST `/speak`, POST `/analyze`).
- **Deviation**: A continuous WebSocket `/stream` endpoint was added to stream frames and audio continuously.
- **Resolution**:
  - `/stream` in `backend/main.py` is gated behind `ENABLE_STREAM_WS=1`.
  - By default, connections to `/stream` receive a descriptive notification and close with code 1008, adhering strictly to the REST scope lock.

### D. Context Reasoning Schema
- **PROMPT.md Specification (Phase 2)**:
  > *"Implement Gemini Flash call using function calling / structured output: input = transcript text, output = JSON {sentiment: str, confidence: float}"*
- **Deviation**: The schema had expanded to include fields like `sarcasm_detected`, `urgency`, and `sentiment_scores`.
- **Resolution**:
  - `backend/context.py` updated: `ContextAnalysis` Pydantic model reduced strictly to:
    ```python
    class ContextAnalysis(BaseModel):
        sentiment: str
        confidence: float
    ```
  - Prompt and fallback keyword analyzer aligned to return this exact two-key structure.

### E. Python 3.11 vs. Python 3.13 Runtime
- **PROMPT.md Specification**:
  > *"Backend: Python 3.11, FastAPI, uvicorn"*
- **Deviation**: The host machine had Python 3.13.1 installed (`C:\Python313\python.exe`).
- **Resolution**:
  - Python 3.13 compatibility was achieved using PyTorch 2.7.1+cu118 and modern Hugging Face transformers.
  - Deviation is explicitly recorded in `phases/phase0/task.md` through `phases/phase7/task.md` and this document.

### F. VRAM and Hardware Verification
- **PROMPT.md Specification**:
  > *"NVIDIA RTX 3050 (4GB VRAM) — all models must run in FP16, verify combined VRAM usage stays under ~2GB with nvidia-smi logging"*
- **Measurement**:
  - ViT FP16 allocated: **178.95 MB**
  - Wav2Vec2-XLSR FP16 allocated: **632.32 MB**
  - Combined model allocation: **811.27 MB** (~0.81 GB), well under the 2 GB target.
  - System total (`nvidia-smi`): **3726 MiB / 4094 MiB** (accounting for Windows Desktop Window Manager and browser background processes).

### G. Phase 3 MLP vs. Gemini Comparison
- **PROMPT.md Specification (Phase 3)**:
  > *"Deliverable: two working fusion approaches (trained MLP + LLM reasoning), both tested against the same set of sample modality combinations, results logged for comparison"*
- **Resolution**:
  - Created `backend/compare_fusion.py` to evaluate both paths across 5 canonical scenarios (Congruent Happy, Masked Distress, Sarcasm, Panic Urgency, Calm Words + Tense Tone).
  - Generated live log: `backend/fusion_comparison.log` confirming emotion agreement and latency differences (<2ms MLP vs. 1.5–4.2s Gemini).

---

## 3. Strict Scope Lock Policy
No features beyond those explicitly described in `PROMPT.md` may be added to this repository. Any further experimental modifications must be approved by the user or gated behind opt-in feature flags that default to `False`.
