"""
tts.py - EmpathAI Phase 4: Text-to-Speech Output Layer
======================================================
Converts the fused emotion result and narration into spoken audio for visually impaired users.

Primary engine: gTTS (Google Text-to-Speech, natural online voice, MP3)
Fallback engine: pyttsx3 (SAPI5 offline Windows speech synthesizer, WAV)
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import threading
from typing import Optional, Tuple

logger = logging.getLogger("empathai.tts")

_PYTTSX3_LOCK = threading.Lock()


def build_spoken_phrase(
    label: str,
    mismatch: bool = False,
    mismatch_kind: str = "none",
    narration: Optional[str] = None,
    allow_long_narration: bool = False,
) -> str:
    """
    Constructs the 1-3 word emotional summary prescribed by PROMPT.md:
    - If mismatch is False: 1-2 word label (e.g. "happy", "calm", "sad", "angry")
    - If mismatch is True: label + "but sounds forced" or "but sounds off" (e.g. "calm, but sounds forced")
    - Long multi-sentence narration is gated behind `allow_long_narration=True` (off by default).
    """
    if allow_long_narration and narration and narration.strip():
        sentences = [s.strip() for s in narration.strip().split(".") if s.strip()]
        return ". ".join(sentences[:2]) + "."

    clean_label = label.lower().strip()
    label_map = {
        "neutral": "calm",
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "fear": "fearful",
        "surprise": "surprised",
        "disgust": "disgusted",
        "calm": "calm",
    }
    short_label = label_map.get(clean_label, clean_label)

    if mismatch:
        if mismatch_kind in ("sarcasm", "passive_aggressive"):
            return f"{short_label}, but sounds off"
        return f"{short_label}, but sounds forced"

    return short_label


def _synthesize_gtts(text: str) -> bytes:
    from gtts import gTTS

    buf = io.BytesIO()
    tts = gTTS(text=text, lang="en", slow=False)
    tts.write_to_fp(buf)
    return buf.getvalue()


def _synthesize_pyttsx3(text: str) -> bytes:
    import pyttsx3

    with _PYTTSX3_LOCK:
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)  # crisp conversational pace
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            with open(tmp_path, "rb") as f:
                data = f.read()
            return data
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass


def synthesize_speech(
    text: str,
    force_offline: bool = False,
) -> Tuple[bytes, str, str]:
    """
    Synthesizes speech from text.
    Returns (audio_bytes, mime_type, engine_used).
    MIME types: "audio/mpeg" (gTTS) or "audio/wav" (pyttsx3).
    """
    clean_text = text.strip() or "No emotion detected."

    if not force_offline:
        try:
            mp3_bytes = _synthesize_gtts(clean_text)
            if len(mp3_bytes) > 0:
                return mp3_bytes, "audio/mpeg", "gTTS"
        except Exception as exc:
            logger.warning(f"gTTS online synthesis failed ({exc}); falling back to pyttsx3.")

    # Offline fallback
    wav_bytes = _synthesize_pyttsx3(clean_text)
    return wav_bytes, "audio/wav", "pyttsx3"


# --------------------------------------------------------------------------- #
# Self-tests
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("Testing build_spoken_phrase():")
    print("  Congruent happy ->", repr(build_spoken_phrase("happy", False)))
    print("  Masked smile ->", repr(build_spoken_phrase("calm", True, "masked_smile")))
    print("  Sarcasm ->", repr(build_spoken_phrase("happy", True, "sarcasm")))
    assert build_spoken_phrase("happy", False) == "happy"
    assert build_spoken_phrase("calm", True, "masked_smile") == "calm, but sounds forced"
    assert build_spoken_phrase("happy", True, "sarcasm") == "happy, but sounds off"

    print("\nTesting online synthesis (gTTS):")
    audio, mime, engine = synthesize_speech("calm", force_offline=False)
    print(f"  Result: {len(audio)} bytes, MIME: {mime}, Engine: {engine}")
    assert len(audio) > 1000 and mime == "audio/mpeg" and engine == "gTTS"

    print("\nTesting offline fallback synthesis (pyttsx3):")
    audio_off, mime_off, engine_off = synthesize_speech("calm, but sounds forced", force_offline=True)
    print(f"  Result: {len(audio_off)} bytes, MIME: {mime_off}, Engine: {engine_off}")
    assert len(audio_off) > 1000 and mime_off == "audio/wav" and engine_off == "pyttsx3"

    print("\nAll tts.py self-tests passed successfully!")
