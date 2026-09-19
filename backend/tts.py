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
) -> str:
    """
    Constructs a concise, screen-reader friendly spoken phrase.
    If full narration is provided, uses it. Otherwise creates a crisp 1-2 clause phrase.
    """
    if narration and narration.strip():
        # Keep it within 2 short sentences
        sentences = [s.strip() for s in narration.strip().split(".") if s.strip()]
        return ". ".join(sentences[:2]) + "."

    adj_map = {
        "angry": "angry",
        "disgust": "disgusted",
        "fear": "fearful",
        "happy": "happy",
        "neutral": "calm and neutral",
        "sad": "sad",
        "surprise": "surprised",
    }
    mood = adj_map.get(label.lower(), label)

    if mismatch:
        if mismatch_kind == "masked_smile":
            return f"Smiling, but tone sounds tense - possibly {mood}."
        elif mismatch_kind == "sarcasm":
            return f"Words sound positive, but delivery suggests sarcasm."

        elif mismatch_kind == "words_vs_tone":
            return f"Words do not match tone; they seem {mood}."
        elif mismatch_kind == "calm_urgent":
            return "Sounding calm, but what they say seems urgent."
        else:
            return f"Conflicting signals; overall they seem {mood}."

    return f"They seem {mood}."


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
    print("  Masked smile ->", repr(build_spoken_phrase("fear", True, "masked_smile")))
    print("  Sarcasm ->", repr(build_spoken_phrase("angry", True, "sarcasm")))

    print("\nTesting online synthesis (gTTS):")
    audio, mime, engine = synthesize_speech("They seem calm and neutral.", force_offline=False)
    print(f"  Result: {len(audio)} bytes, MIME: {mime}, Engine: {engine}")
    assert len(audio) > 1000 and mime == "audio/mpeg" and engine == "gTTS"

    print("\nTesting offline fallback synthesis (pyttsx3):")
    audio_off, mime_off, engine_off = synthesize_speech("Smiling, but tone sounds tense.", force_offline=True)
    print(f"  Result: {len(audio_off)} bytes, MIME: {mime_off}, Engine: {engine_off}")
    assert len(audio_off) > 1000 and mime_off == "audio/wav" and engine_off == "pyttsx3"

    print("\nAll tts.py self-tests passed successfully!")
