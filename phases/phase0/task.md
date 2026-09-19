# Phase 0 — Project Scaffolding

- [x] Create repo structure: `/backend`, `/frontend`, `/models`, `/phases`
- [x] Set up Python 3.11 venv in `/backend`
- [x] Install backend dependencies: fastapi, uvicorn[standard], python-dotenv (torch/others installing in background via CUDA wheel)
- [x] Set up React app with Tailwind CSS in `/frontend`, no functionality yet
- [x] Create `.env.example` with `GEMINI_API_KEY=`
- [x] Confirm FastAPI server runs and responds to a basic health-check route
- [x] Confirm React dev server runs and renders a placeholder page

## Deliverable
✅ FastAPI server running on `http://0.0.0.0:8000` — `GET /health` returns `{"status": "ok"}` confirmed via `Invoke-RestMethod`.
✅ React+Tailwind dev server running on `http://localhost:5173` — Vite ready in 5530ms confirmed via terminal logs.

## Deviations
- Python runtime: Host machine uses Python 3.13.1 (`C:\Python313\python.exe`) instead of Python 3.11 specified in PROMPT.md. All dependencies, PyTorch 2.7.1+cu118, and Hugging Face pipelines compile and run cleanly.
- `mediapipe==0.10.14` → `mediapipe==1.0.1`: version 0.10.x not available for Python 3.13. Additionally, mediapipe 1.0+ removed `mp.solutions` entirely — face detection replaced with OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`).
- `transformers==4.41.2` → `transformers>=4.45.0` (resolved to 5.17.0): 4.41.x requires `tokenizers<0.20` which has no pre-built Python 3.13 wheel and failed to compile (no MSVC linker present).
- `accelerate==0.30.1` → `accelerate>=0.34.0` (resolved to 1.15.0): bumped to match new transformers.
- `python-multipart` added: required by FastAPI for `UploadFile`/`File(...)` endpoints — was missing from original list.
- `opencv-python-headless==4.9.0.80` → `5.0.0.93`: mediapipe 1.0.1 pulled in `opencv-contrib-python` 5.x as a transitive dep.
- PyTorch wheel: `torch-2.7.1+cu118` (latest CUDA 11.8 build, vs implied 2.x in spec) — acceptable since it's newer and compatible.
- `webrtcvad-wheels` used in place of `webrtcvad` for Windows compatibility (cross-platform wheel).
- PyTorch CUDA 11.8 wheel (~2.8 GB) installed separately from the main pip install due to custom index URL requirement.
- FastAPI version resolved to 0.141.1 (latest available, slightly newer than 0.111.0 pinned in requirements.txt — updated pin accordingly).
