"""
transcript.py
==============
Speech-to-text via faster-whisper on CPU (int8).

Fixes applied:
1. LANGUAGE HALLUCINATIONS: Enforces `language="en"` by default to prevent
   hallucinating non-English languages on room noise or quiet breath.
2. NOISE HALLUCINATIONS:
   - greedy decoding (`temperature=0.0`)
   - `condition_on_previous_text=False` (prevents repetition loops)
   - built-in Silero VAD (`vad_filter=True`)
   - post-hoc filtering on `no_speech_prob > 0.6` and `avg_logprob < -1.0`
3. Provides both `Transcriber` class and `transcribe_audio` drop-in helper.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    text: str
    start: float
    end: float
    avg_logprob: float
    no_speech_prob: float


@dataclass
class TranscriptResult:
    text: str
    segments: List[TranscriptSegment] = field(default_factory=list)
    language: str = "en"


@dataclass
class TranscriberConfig:
    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    language: Optional[str] = "en"        # force English; eliminates language guessing on noise
    beam_size: int = 5
    temperature: float = 0.0              # greedy decoding (minimal hallucinations)
    condition_on_previous_text: bool = False
    vad_filter: bool = True               # built-in Silero VAD
    no_speech_threshold: float = 0.6      # drop segments with no_speech_prob > 0.6
    avg_logprob_threshold: float = -1.0   # drop segments with low confidence


class Transcriber:
    """Stateless per call — safe to share across concurrent requests."""

    def __init__(self, cfg: TranscriberConfig = TranscriberConfig()):
        self.cfg = cfg
        logger.info("Loading faster-whisper '%s' on %s (%s)...", cfg.model_size, cfg.device, cfg.compute_type)
        self.model = WhisperModel(cfg.model_size, device=cfg.device, compute_type=cfg.compute_type)
        logger.info("Faster-whisper loaded.")

    def transcribe(self, audio: Union[bytes, bytearray, np.ndarray, str]) -> TranscriptResult:
        if isinstance(audio, (bytes, bytearray)):
            source = io.BytesIO(bytes(audio))
        else:
            source = audio

        segments_iter, info = self.model.transcribe(
            source,
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            temperature=self.cfg.temperature,
            condition_on_previous_text=self.cfg.condition_on_previous_text,
            vad_filter=self.cfg.vad_filter,
        )

        kept: List[TranscriptSegment] = []
        for seg in segments_iter:
            if seg.no_speech_prob > self.cfg.no_speech_threshold:
                continue
            if seg.avg_logprob < self.cfg.avg_logprob_threshold:
                continue
            text = seg.text.strip()
            if not text:
                continue
            kept.append(TranscriptSegment(
                text=text, start=seg.start, end=seg.end,
                avg_logprob=seg.avg_logprob, no_speech_prob=seg.no_speech_prob,
            ))

        full_text = " ".join(s.text for s in kept).strip()
        language = self.cfg.language or getattr(info, "language", "en")
        return TranscriptResult(text=full_text, segments=kept, language=language)


# ── Global singleton & drop-in transcription helper ───────────────────────────

_GLOBAL_TRANSCRIBER: Optional[Transcriber] = None

def transcribe_audio(audio_bytes: bytes) -> dict:
    """Drop-in functional entrypoint for main.py / FastAPI."""
    global _GLOBAL_TRANSCRIBER
    if _GLOBAL_TRANSCRIBER is None:
        _GLOBAL_TRANSCRIBER = Transcriber()
    res = _GLOBAL_TRANSCRIBER.transcribe(audio_bytes)
    return {
        "transcript": res.text,
        "language": res.language,
        "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in res.segments],
    }


if __name__ == "__main__":
    class _FakeSeg:
        def __init__(self, text, start, end, avg_logprob, no_speech_prob):
            self.text, self.start, self.end = text, start, end
            self.avg_logprob, self.no_speech_prob = avg_logprob, no_speech_prob

    class _FakeInfo:
        language = "en"

    class _FakeModel:
        def __init__(self, segments):
            self._segments = segments

        def transcribe(self, source, **kwargs):
            return iter(self._segments), _FakeInfo()

    cfg = TranscriberConfig()
    t = Transcriber.__new__(Transcriber)
    t.cfg = cfg
    t.model = _FakeModel([
        _FakeSeg("hello there", 0.0, 1.0, avg_logprob=-0.3, no_speech_prob=0.1),
        _FakeSeg("thank you", 1.0, 2.0, avg_logprob=-0.4, no_speech_prob=0.05),
        _FakeSeg("...", 2.0, 2.5, avg_logprob=-0.5, no_speech_prob=0.9),
        _FakeSeg("garbled noise text", 2.5, 3.5, avg_logprob=-1.8, no_speech_prob=0.2),
        _FakeSeg("  ", 3.5, 4.0, avg_logprob=-0.2, no_speech_prob=0.1),
    ])
    result = t.transcribe(b"irrelevant-in-this-fake")
    assert result.text == "hello there thank you", result.text
    assert len(result.segments) == 2, len(result.segments)
    assert result.language == "en"
    print("transcript.py filtering self-test: PASS ->", repr(result.text))

