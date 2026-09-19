"""
EmpathAI — FastAPI backend.
Phase 0: /health
Phase 1:
  - WS   /stream       — continuous real-time analysis
                         • webcam: sampled every FACE_INTERVAL_SEC
                         • audio : VAD-based — analyses complete utterances, not fixed chunks
  - POST /face-emotion — image bytes → emotion (for frontend)
  - POST /speech-emotion — audio bytes → emotion (for frontend)
  - POST /transcript   — audio bytes → transcript (for frontend)
"""

import asyncio
import io
import json
import logging
import os
import threading
import time

import cv2
import numpy as np

from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, Response

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FACE_INTERVAL_SEC = 0.5  # how often to sample a webcam frame

app = FastAPI(
    title="EmpathAI",
    description="Multimodal emotion recognition for visually impaired users.",
    version="0.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Phase 0 ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health_check():
    return {"status": "ok"}


# ── Webcam helper ─────────────────────────────────────────────────────────────

def _grab_frame(camera_index: int = 0) -> bytes | None:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None
    try:
        ret, frame = cap.read()
        if not ret:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None
    finally:
        cap.release()


# ── Phase 1 — Real-time WebSocket stream ──────────────────────────────────────

@app.websocket("/stream")
async def realtime_stream(websocket: WebSocket, camera: int = 0):
    """
    Continuous real-time multimodal analysis over WebSocket (Optional Extra).
    Disabled by default per PROMPT.md scope lock.
    Enable by setting environment variable ENABLE_STREAM_WS=1.
    """
    if not os.getenv("ENABLE_STREAM_WS", "false").lower() in ("true", "1", "yes"):
        await websocket.accept()
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "/stream is an optional extra and is disabled by default per PROMPT.md scope lock. Set ENABLE_STREAM_WS=1 to enable."
        }))
        await websocket.close(code=1008, reason="/stream disabled by default per PROMPT.md. Set ENABLE_STREAM_WS=1 to enable.")
        return

    await websocket.accept()
    logger.info("WS /stream connected")

    from face_emotion import predict_face_emotion
    from speech_emotion import predict_speech_emotion
    from transcript import transcribe_audio
    from audio_capture import capture_speech_segments
    from denoise import denoise_wav_bytes

    loop = asyncio.get_event_loop()
    stop_event = threading.Event()

    # Shared state pushed from background threads
    result_queue: asyncio.Queue = asyncio.Queue()

    # ── Face worker ───────────────────────────────────────────────────────
    async def face_worker():
        if camera < 0:
            logger.info("WS /stream: audio-only mode requested (camera < 0); skipping webcam capture.")
            return
        while not stop_event.is_set():
            t0 = time.perf_counter()
            frame_bytes = await loop.run_in_executor(None, _grab_frame, camera)
            if frame_bytes:
                try:
                    face = await loop.run_in_executor(None, predict_face_emotion, frame_bytes)
                    face["latency_ms"] = round((time.perf_counter() - t0) * 1000)
                    await result_queue.put({"type": "face", "data": face, "ts": time.time()})
                except Exception as e:
                    logger.error("Face worker error: %s", e)
                    await result_queue.put({"type": "face", "data": {"error": str(e)}, "ts": time.time()})
            await asyncio.sleep(FACE_INTERVAL_SEC)

    # ── Audio worker ──────────────────────────────────────────────────────
    def audio_worker():
        """Runs in a thread — VAD blocks until each utterance ends."""
        try:
            for wav_bytes in capture_speech_segments(stop_event=stop_event):
                if stop_event.is_set():
                    break
                wav_bytes = denoise_wav_bytes(wav_bytes)
                t0 = time.perf_counter()
                try:
                    speech  = predict_speech_emotion(wav_bytes)
                    tx      = transcribe_audio(wav_bytes)
                    latency = round((time.perf_counter() - t0) * 1000)
                    asyncio.run_coroutine_threadsafe(
                        result_queue.put({
                            "type": "audio",
                            "speech": {**speech, "latency_ms": latency},
                            "transcript": {**tx, "latency_ms": latency},
                            "ts": time.time(),
                        }),
                        loop,
                    )
                except Exception as e:
                    logger.error("Audio worker error: %s", e)
                    asyncio.run_coroutine_threadsafe(
                        result_queue.put({"type": "audio", "error": str(e), "ts": time.time()}),
                        loop,
                    )
        except Exception as exc:
            logger.error("Audio worker fatal error: %s", exc)

    audio_thread = threading.Thread(target=audio_worker, daemon=True)
    audio_thread.start()

    face_task = asyncio.create_task(face_worker())

    async def client_listener():
        try:
            while not stop_event.is_set():
                msg = await websocket.receive_text()
                if msg.strip().lower() in ("stop", "quit", "disconnect"):
                    stop_event.set()
                    break
        except (WebSocketDisconnect, asyncio.CancelledError):
            stop_event.set()
        except Exception as exc:
            logger.debug("WS client listener error: %s", exc)
            stop_event.set()

    listener_task = asyncio.create_task(client_listener())

    try:
        while not stop_event.is_set():
            try:
                packet = await asyncio.wait_for(result_queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue

            await websocket.send_text(json.dumps(packet))
            ptype = packet.get("type", "unknown")
            if ptype == "face":
                data = packet.get("data", {})
                emo = data.get("emotion", data.get("error", "?"))
                logger.info("→ face: %s", emo)
            elif ptype == "audio":
                speech = packet.get("speech", {})
                emo = speech.get("emotion", packet.get("error", "?"))
                logger.info("→ audio: %s", emo)

    except WebSocketDisconnect:
        logger.info("WS /stream disconnected by client")
    except Exception as exc:
        logger.exception("WS /stream error: %s", exc)
    finally:
        stop_event.set()
        face_task.cancel()
        listener_task.cancel()
        logger.info("WS /stream cleaned up")


# ── Phase 1 — Single-shot POST endpoints (frontend sends bytes) ───────────────

@app.post("/face-emotion", tags=["phase1"])
async def face_emotion(image: UploadFile = File(...)):
    try:
        from face_emotion import predict_face_emotion
        return predict_face_emotion(await image.read())
    except Exception as exc:
        logger.exception("face-emotion error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/speech-emotion", tags=["phase1"])
async def speech_emotion(audio: UploadFile = File(...)):
    try:
        from denoise import denoise_wav_bytes
        from speech_emotion import predict_speech_emotion
        raw_bytes = await audio.read()
        cleaned_bytes = denoise_wav_bytes(raw_bytes)
        return predict_speech_emotion(cleaned_bytes)
    except Exception as exc:
        logger.exception("speech-emotion error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/transcript", tags=["phase1"])
async def transcript(audio: UploadFile = File(...)):
    try:
        from denoise import denoise_wav_bytes
        from transcript import transcribe_audio
        raw_bytes = await audio.read()
        cleaned_bytes = denoise_wav_bytes(raw_bytes)
        return transcribe_audio(cleaned_bytes)
    except Exception as exc:
        logger.exception("transcript error")
        raise HTTPException(status_code=500, detail=str(exc))


class ContextRequest(BaseModel):
    text: str


@app.post("/context", tags=["phase2"])
async def context(req: ContextRequest):
    try:
        from context import analyze_context
        return analyze_context(req.text)
    except Exception as exc:
        logger.exception("context error")
        raise HTTPException(status_code=500, detail=str(exc))


class FuseRequest(BaseModel):
    face: Optional[dict] = None
    speech: Optional[dict] = None
    context: Optional[dict] = None
    mode: str = "fast"


@app.post("/fuse", tags=["phase3"])
async def fuse_endpoint(req: FuseRequest):
    try:
        from fusion import fuse
        res = fuse(face=req.face, speech=req.speech, context=req.context, mode=req.mode)
        return res.to_dict()
    except Exception as exc:
        logger.exception("fusion error")
        raise HTTPException(status_code=500, detail=str(exc))


class SpeakRequest(BaseModel):
    text: Optional[str] = None
    label: Optional[str] = None
    mismatch: bool = False
    mismatch_kind: str = "none"
    narration: Optional[str] = None
    force_offline: bool = False


@app.post("/speak", tags=["phase4"])
async def speak_endpoint(req: SpeakRequest):
    try:
        from tts import build_spoken_phrase, synthesize_speech
        phrase = req.text or build_spoken_phrase(
            label=req.label or "neutral",
            mismatch=req.mismatch,
            mismatch_kind=req.mismatch_kind,
            narration=req.narration,
        )
        audio_bytes, mime_type, engine_used = synthesize_speech(
            phrase, force_offline=req.force_offline
        )
        return Response(
            content=audio_bytes,
            media_type=mime_type,
            headers={
                "X-Engine-Used": engine_used,
                "X-Spoken-Phrase": phrase,
            },
        )
    except Exception as exc:
        logger.exception("tts error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/analyze", tags=["phase5"])
async def analyze_endpoint(
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    mode: str = Form("fast"),
    generate_tts: bool = Form(True),
    face_hint: Optional[str] = Form(None),
):
    try:
        import json
        from pipeline import analyze_multimodal
        from denoise import denoise_wav_bytes

        img_bytes = (await image.read()) if image is not None else None
        aud_bytes = (await audio.read()) if audio is not None else None
        if aud_bytes is not None:
            aud_bytes = denoise_wav_bytes(aud_bytes)
        vid_bytes = (await video.read()) if video is not None else None

        parsed_hint = None
        if face_hint:
            try:
                parsed_hint = json.loads(face_hint)
            except Exception:
                pass

        return analyze_multimodal(
            image_bytes=img_bytes,
            audio_bytes=aud_bytes,
            video_bytes=vid_bytes,
            mode=mode,
            generate_tts=generate_tts,
            face_hint=parsed_hint,
        )
    except Exception as exc:
        logger.exception("analyze error")
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":

    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)



