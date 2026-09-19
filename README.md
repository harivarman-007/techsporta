# EmpathAI

> **Real-Time Multimodal Emotion Recognition & Mismatch Detection for Visually Impaired Users**

EmpathAI is an assistive AI system designed to help blind and low-vision individuals perceive the subtle, nonverbal emotional cues of people they interact with. By independently analyzing **facial expressions (ViT + MediaPipe Face Mesh)**, **vocal tone (Wav2Vec2-XLSR)**, and **spoken context (faster-whisper + Gemini Flash)**, EmpathAI fuses these signals to detect emotional congruence or flag critical **channel contradictions** (such as emotional masking, faked smiles, sarcasm, or passive-aggressive dissonance).

---

## Key Features

1. **Facial Expression & Muscle Tracking**:
   - 478 3D MediaPipe face mesh landmarks + 52 FACS (Facial Action Coding System) muscle blendshapes.
   - Per-user **Neutral Calibration** (`[C]` key) to eliminate resting-face bias (RBF).
   - Tight landmark cropping to avoid FER2013 distribution shifts.
   - Temporal smoothing (EMA + hysteresis) to eliminate frame jitter.

2. **Acoustic Vocal Tone Analysis**:
   - `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` running in FP16 on CUDA.
   - RMS and Peak loudness normalization (target -20 dBFS) with silence gating to prevent flat uniform probabilities on quiet laptop microphones.
   - 100ms hardware audio buffering (`DEFAULT_BLOCKSIZE = 1600`) to eliminate Windows WASAPI input overflows.

3. **Hallucination-Proof Transcription & Context Reasoning**:
   - `faster-whisper` on CPU with `int8` quantization (<150ms per utterance, 0 MB GPU VRAM).
   - Fixed `language="en"` to completely eliminate auto-detect language hallucinations.
   - Google Gemini Flash structured output classifying sentiment, urgency level, and sarcasm flags.

4. **Dual-Path Multimodal Fusion Layer**:
   - **Fast Edge Path (<1ms, CPU)**: A 3-layer PyTorch MLP (`MLPFusionHead`) trained on 40,000 synthetic scenarios that outputs the unified emotion label, mismatch flag, and screen-reader friendly template narration.
   - **Rich Reasoning Path (Gemini Flash Reconciler)**: Generates rich natural-language spoken synthesis explaining subtle nonverbal conflicts with hedged, respectful language.

5. **Spoken Output Layer (TTS)**:
   - Online: Natural Google TTS (`gTTS`) streaming MP3.
   - Offline Fallback: Windows SAPI5 (`pyttsx3`) streaming WAV without internet connection.

6. **Interactive Accessible Frontend**:
   - React + Tailwind CSS with dark-mode aesthetic, large accessible indicators, spacebar recording shortcuts, and live latency diagnostics.

---

## Literature Survey Limitations Addressed

1. **High Compute Cost & Hardware Inaccessibility**:
   - *Problem*: Conventional multimodal architectures require massive GPU clusters or dual 24GB GPUs.
   - *EmpathAI Solution*: Optimized to run entirely on a laptop GPU (NVIDIA RTX 3050 4GB). ViT and Wav2Vec2 run in FP16, Whisper runs in int8 on CPU, and the Fusion Head is a micro-MLP (<1ms). Total system VRAM stays below **2.3 GB**.

2. **Limited Contextual Understanding**:
   - *Problem*: Traditional speech/face systems treat utterances in isolation, mistaking sarcastic phrases like *"Oh wonderful, another flat tire!"* for genuine happiness.
   - *EmpathAI Solution*: Incorporates a dedicated Spoken Context Reasoning Layer using Gemini Flash to evaluate verbal nuance, semantic irony, and urgency.

3. **Channel Conflicts & Emotional Masking Neglect**:
   - *Problem*: Standard fusion methods average probabilities, washing out conflicting signals.
   - *EmpathAI Solution*: EmpathAI treats channels as independent witnesses. The Fusion Layer explicitly evaluates pairwise cross-channel conflict terms and raises a dedicated `mismatch_detected` warning whenever words, facial muscles, or vocal pitch contradict each other.

---

## System Architecture

```
                               ┌───────────────────────────┐
                               │       User Inputs         │
                               └─────────────┬─────────────┘
                                             │
             ┌───────────────────────────────┼───────────────────────────────┐
             │                               │                               │
             ▼                               ▼                               ▼
    [Webcam Video Stream]           [Microphone Audio]              [Audio Utterance]
             │                               │                               │
             ▼                               ▼                               ▼
     MediaPipe + ViT                   Wav2Vec2-XLSR                  faster-whisper
   Face Emotion (7-class)          Speech Emotion (8-class)            (CPU int8)
    + 52 FACS Blendshapes            + Loudness Norm                         │
             │                               │                               ▼
             │                               │                          Gemini Flash
             │                               │                     Context Sentiment & Sarcasm
             │                               │                               │
             └───────────────────────┬───────┴───────────────────────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │    Phase 3 Fusion Layer     │
                      │  • 3-Layer PyTorch MLP      │
                      │  • Gemini Flash Reconciler  │
                      └──────────────┬──────────────┘
                                     │
                         [Fused Label + Mismatch]
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │     Phase 4 Output Layer    │
                      │  gTTS / pyttsx3 Synthesizer │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │   Spoken Audio + Frontend   │
                      │   Accessible Screen Reader  │
                      └─────────────────────────────┘
```

---

## Quickstart & How to Run

### Prerequisites
- Windows 10/11
- Python 3.10+ (tested on Python 3.13)
- Node.js 18+
- NVIDIA GPU (RTX 3050 or higher recommended) with CUDA drivers

### 1. Backend Setup & Run
```powershell
cd c:\projects\hacksporta\backend

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# (Optional) Verify or add your Gemini API Key in .env
# GEMINI_API_KEY=AIzaSy...

# Start the FastAPI server
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
* Backend API Documentation: `http://localhost:8000/docs`
* Network IP: `http://10.178.236.88:8000`

### 2. Frontend Setup & Run
```powershell
cd c:\projects\hacksporta\frontend

# Install dependencies (if not already installed)
npm install

# Start development server
npm run dev
```
* Local Web UI: `http://localhost:5173`
* Network Access: `http://10.178.236.88:5173` (accessible on mobile/tablet via local Wi-Fi)

### 3. Real-Time Face & Emotion Calibration GUI (Standalone Tool)
To inspect your 3D facial mesh, test FACS action units, and perform interactive neutral resting face calibration:
```powershell
cd c:\projects\hacksporta\backend
.\venv\Scripts\python live_face_view.py
```
* Press **`C`** to calibrate your neutral baseline (hold relaxed face for 2 seconds).
* Press **`M`** to toggle 468-point 3D Face Mesh overlay.
* Press **`B`** to toggle bounding box.
* Press **`Q`** or **`ESC`** to exit.

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Server health check |
| `POST` | `/face-emotion` | Image file $\rightarrow$ 7-class emotion probabilities & FACS deltas |
| `POST` | `/speech-emotion`| Audio file $\rightarrow$ 8-class vocal tone probabilities & confidence |
| `POST` | `/transcript` | Audio file $\rightarrow$ English speech transcript |
| `POST` | `/context` | Text transcript $\rightarrow$ Gemini Flash sentiment, urgency, and sarcasm |
| `POST` | `/fuse` | Multimodal vectors $\rightarrow$ PyTorch MLP / Gemini fused label + mismatch |
| `POST` | `/speak` | Fused label / text $\rightarrow$ Spoken MP3/WAV audio stream |
| `POST` | `/analyze` | Single end-to-end endpoint (video/image + audio $\rightarrow$ full analysis + TTS audio + latency breakdown) |
| `WS` | `/stream` | WebSocket real-time webcam & microphone streaming |

---

## Benchmark & Accuracy Summary

Run the Phase 7 benchmark suite:
```powershell
cd c:\projects\hacksporta\backend
.\venv\Scripts\python test_phase7.py
```

### Results
- **Test Cases Passed**: **5 / 5 (100.0%)**
- **MLP Mismatch Detection Precision**: **93.8%**
- **MLP Mismatch Detection Recall**: **92.28%**
- **Average Pipeline Latencies (Steady-State)**:
  - Face Inference: **~208 ms**
  - Vocal Tone: **~78 ms**
  - Speech-to-Text: **~289 ms**
  - Context Reasoning: **~350 ms**
  - Fusion Layer: **< 1 ms**
  - TTS Output: **~500 ms**
  - **Total Pipeline**: **~1.08 seconds** (well within the 5–8s target)
#   t e c h s p o r t a  
 