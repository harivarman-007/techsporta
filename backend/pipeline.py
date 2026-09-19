"""
pipeline.py - EmpathAI Phase 5: End-to-End Multimodal Analysis Pipeline
========================================================================
Chains all modalities together:
  Video/Image -> Face Emotion (ViT + MediaPipe FACS)
  Video/Audio -> Speech Emotion (Wav2Vec2-XLSR)
  Video/Audio -> Transcript (faster-whisper)
  Transcript  -> Context Reasoning (Gemini Flash / Heuristics)
  All Cues    -> Fusion Layer (MLPFusionHead + Gemini Reconciler)
  Summary     -> TTS Output (gTTS / pyttsx3)

Provides per-stage latency tracking and structured error reporting.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("empathai.pipeline")


def extract_media_from_video(video_bytes: bytes) -> Tuple[Optional[bytes], Optional[bytes]]:
    """
    Extracts a representative face image (JPEG) and a 16kHz mono audio track (WAV)
    from video bytes.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        video_path = f.name

    image_bytes = None
    audio_bytes = None

    try:
        # 1. Extract middle frame as JPEG
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        target_frame = max(0, total_frames // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            _, buf = cv2.imencode(".jpg", frame)
            image_bytes = buf.tobytes()

        # 2. Extract audio track as 16kHz mono WAV using ffmpeg
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            wav_tmp = video_path + ".wav"
            cmd = [
                ffmpeg_exe, "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                wav_tmp,
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
            if result.returncode == 0 and os.path.exists(wav_tmp):
                with open(wav_tmp, "rb") as wf:
                    audio_bytes = wf.read()
                try:
                    os.remove(wav_tmp)
                except Exception:
                    pass
        except Exception as exc:
            logger.warning(f"Failed to extract audio track with ffmpeg: {exc}")

    finally:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass

    return image_bytes, audio_bytes


def analyze_multimodal(
    image_bytes: Optional[bytes] = None,
    audio_bytes: Optional[bytes] = None,
    video_bytes: Optional[bytes] = None,
    mode: str = "fast",
    generate_tts: bool = True,
    force_offline_tts: bool = False,
) -> Dict[str, Any]:
    """
    Executes the complete multimodal pipeline with fine-grained latency tracking.
    """
    t_start = time.perf_counter()
    latencies: Dict[str, float] = {}
    errors: List[str] = []

    # If video was provided, extract components if not explicitly given
    if video_bytes and len(video_bytes) > 0:
        extracted_img, extracted_aud = extract_media_from_video(video_bytes)
        if image_bytes is None:
            image_bytes = extracted_img
        if audio_bytes is None:
            audio_bytes = extracted_aud

    # Apply noise suppression if audio is provided
    if audio_bytes and len(audio_bytes) > 0:
        try:
            from denoise import denoise_wav_bytes
            audio_bytes = denoise_wav_bytes(audio_bytes)
        except Exception as exc:
            logger.warning("Pipeline audio denoise fallback: %s", exc)

    # 1. Face Emotion Inference
    face_res = None
    if image_bytes and len(image_bytes) > 0:
        t0 = time.perf_counter()
        try:
            from face_emotion import predict_face_emotion
            face_res = predict_face_emotion(image_bytes)
            latencies["face"] = round((time.perf_counter() - t0) * 1e3, 2)
        except Exception as exc:
            logger.exception("Face emotion failed")
            errors.append(f"Face emotion error: {exc}")
            latencies["face"] = round((time.perf_counter() - t0) * 1e3, 2)

    # 2. Speech Emotion Inference
    speech_res = None
    if audio_bytes and len(audio_bytes) > 0:
        t0 = time.perf_counter()
        try:
            from speech_emotion import predict_speech_emotion
            speech_res = predict_speech_emotion(audio_bytes)
            latencies["speech"] = round((time.perf_counter() - t0) * 1e3, 2)
        except Exception as exc:
            logger.exception("Speech emotion failed")
            errors.append(f"Speech emotion error: {exc}")
            latencies["speech"] = round((time.perf_counter() - t0) * 1e3, 2)

    # 3. Audio Transcription
    transcript_res = None
    transcript_text = ""
    if audio_bytes and len(audio_bytes) > 0:
        t0 = time.perf_counter()
        try:
            from transcript import transcribe_audio
            transcript_res = transcribe_audio(audio_bytes)
            transcript_text = transcript_res.get("transcript", "")
            latencies["transcript"] = round((time.perf_counter() - t0) * 1e3, 2)
        except Exception as exc:
            logger.exception("Transcription failed")
            errors.append(f"Transcription error: {exc}")
            latencies["transcript"] = round((time.perf_counter() - t0) * 1e3, 2)

    # 4. Context Reasoning
    context_res = None
    if transcript_text:
        t0 = time.perf_counter()
        try:
            from context import analyze_context
            context_res = analyze_context(transcript_text)
            latencies["context"] = round((time.perf_counter() - t0) * 1e3, 2)
        except Exception as exc:
            logger.exception("Context reasoning failed")
            errors.append(f"Context reasoning error: {exc}")
            latencies["context"] = round((time.perf_counter() - t0) * 1e3, 2)

    # 5. Multimodal Fusion
    t0 = time.perf_counter()
    fusion_dict = {}
    try:
        from fusion import fuse
        fusion_res = fuse(
            face=face_res,
            speech=speech_res,
            context=context_res,
            mode=mode,
        )
        fusion_dict = fusion_res.to_dict()
        latencies["fusion"] = round((time.perf_counter() - t0) * 1e3, 2)
    except Exception as exc:
        logger.exception("Fusion failed")
        errors.append(f"Fusion error: {exc}")
        latencies["fusion"] = round((time.perf_counter() - t0) * 1e3, 2)
        # Fallback default
        fusion_dict = {
            "label": "neutral",
            "confidence": 0.0,
            "mismatch": False,
            "mismatch_kind": "none",
            "urgency": "low",
            "narration": "Emotion could not be fused.",
            "narration_source": "template",
        }

    # 6. Text-to-Speech Output
    audio_base64 = None
    tts_engine = None
    from tts import build_spoken_phrase, synthesize_speech
    spoken_phrase = build_spoken_phrase(
        label=fusion_dict.get("label", "neutral"),
        mismatch=fusion_dict.get("mismatch", False),
        mismatch_kind=fusion_dict.get("mismatch_kind", "none"),
        narration=fusion_dict.get("narration", ""),
    )
    if generate_tts and spoken_phrase:
        t0 = time.perf_counter()
        try:
            tts_bytes, mime_type, tts_engine = synthesize_speech(
                spoken_phrase, force_offline=force_offline_tts
            )
            b64 = base64.b64encode(tts_bytes).decode("ascii")
            audio_base64 = f"data:{mime_type};base64,{b64}"
            latencies["tts"] = round((time.perf_counter() - t0) * 1e3, 2)
        except Exception as exc:
            logger.exception("TTS synthesis failed")
            errors.append(f"TTS synthesis error: {exc}")
            latencies["tts"] = round((time.perf_counter() - t0) * 1e3, 2)

    latencies["total"] = round((time.perf_counter() - t_start) * 1e3, 2)

    return {
        "status": "success" if not errors else "partial_success",
        "primary_emotion": fusion_dict.get("label", "neutral"),
        "confidence": fusion_dict.get("confidence", 0.0),
        "mismatch_detected": fusion_dict.get("mismatch", False),
        "mismatch_kind": fusion_dict.get("mismatch_kind", "none"),
        "urgency": fusion_dict.get("urgency", "low"),
        "spoken_summary": spoken_phrase,
        "audio_base64": audio_base64,
        "tts_engine": tts_engine,
        "intermediate_results": {
            "face": face_res,
            "speech": speech_res,
            "transcript": transcript_res,
            "context": context_res,
            "fusion": fusion_dict,
        },
        "latencies_ms": latencies,
        "errors": errors,
    }


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("Testing pipeline.py with synthetic inputs:")
    # Create a synthetic 16kHz mono WAV
    sr = 16000
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    sig = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    pcm = (sig * 32767).astype(np.int16)
    buf = io.BytesIO()
    import wave
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    test_wav = buf.getvalue()

    # Create a dummy image
    dummy_img = np.full((300, 300, 3), 128, dtype=np.uint8)
    _, img_buf = cv2.imencode(".jpg", dummy_img)
    test_img = img_buf.tobytes()

    res = analyze_multimodal(image_bytes=test_img, audio_bytes=test_wav, mode="fast", generate_tts=True)
    print("\nPipeline Result:")
    print("  Status:", res["status"])
    print("  Primary Emotion:", res["primary_emotion"])
    print("  Mismatch Detected:", res["mismatch_detected"])
    print("  Spoken Summary:", res["spoken_summary"])
    print("  TTS Generated:", bool(res["audio_base64"]))
    print("  Latencies (ms):", res["latencies_ms"])
    if res["errors"]:
        print("  Errors / Warnings:", res["errors"])

    print("\nPipeline test completed successfully.")
