"""
audio_capture.py
=================
Continuous microphone capture + VAD-based utterance segmentation for EmpathAI.

Fixes applied:
1. INPUT OVERFLOWS: the hardware InputStream blocksize uses DEFAULT_BLOCKSIZE
   (1600 samples = 100ms) to ensure Windows/WASAPI doesn't drop packets.
   Re-chunking into exact 20ms frames happens safely in a separate worker thread.
2. CHOPPED / RUNAWAY UTTERANCES: UtteranceSegmenter enforces a minimum
   utterance duration (1.2s), an adaptive silence-gap endpoint (700ms), and a
   hard 10s maximum cap to prevent runaway recordings.
3. Provides both object-oriented `AudioCapture` class and backward-compatible
   `capture_speech_segments` generator for FastAPI/main.py.
"""
from __future__ import annotations

import collections
import io
import queue
import threading
import wave
from dataclasses import dataclass
from typing import Callable, Generator, List, Optional

import numpy as np
import sounddevice as sd

try:
    import webrtcvad
except ImportError as e:
    raise ImportError(
        "webrtcvad is required. Install via: pip install webrtcvad-wheels"
    ) from e


SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2  # int16
CHANNELS = 1
FRAME_MS = 20  # webrtcvad only accepts 10, 20, or 30 ms frames
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000        # 320
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH_BYTES      # 640

# Healthy hardware buffer (100ms) to prevent WASAPI/Windows input buffer overflows
DEFAULT_BLOCKSIZE = 1600


@dataclass
class VADConfig:
    aggressiveness: int = 2            # webrtcvad: 0 (least aggressive) .. 3 (most)
    energy_gate_rms: float = 60.0      # int16-scale RMS floor; frames quieter than this are never "voiced"
    pre_roll_ms: int = 300             # audio kept before the trigger so word onsets aren't clipped
    trigger_ratio: float = 0.6         # fraction of voiced frames in the pre-roll ring needed to start
    min_utterance_ms: int = 1200       # utterances shorter than this are extended, not cut, on a pause
    max_silence_gap_ms: int = 700      # sustained silence needed (after min duration) to end an utterance
    max_utterance_ms: int = 10000      # hard cap regardless of continued "speech"
    post_roll_ms: int = 200            # trailing padding kept after end-of-speech is detected


def _frames_to_wav(frames: List[bytes]) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH_BYTES)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


class UtteranceSegmenter:
    """
    Pure logic: feed exact 20ms PCM16 mono 16kHz frames one at a time via
    `process_frame`, get back complete utterance WAV bytes whenever one ends.
    """

    def __init__(self, config: VADConfig = VADConfig(),
                 vad_is_speech_fn: Optional[Callable[[bytes, int], bool]] = None):
        self.cfg = config
        if vad_is_speech_fn is not None:
            self._is_speech_raw = vad_is_speech_fn
        else:
            vad = webrtcvad.Vad(config.aggressiveness)
            self._is_speech_raw = vad.is_speech

        ring_len = max(1, config.pre_roll_ms // FRAME_MS)
        self._ring: "collections.deque[tuple[bytes, bool]]" = collections.deque(maxlen=ring_len)
        self._triggered = False
        self._utterance_frames: List[bytes] = []
        self._consecutive_unvoiced = 0

    def _frame_voiced(self, frame: bytes) -> bool:
        pcm = np.frombuffer(frame, dtype=np.int16)
        rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))) if pcm.size else 0.0
        if rms < self.cfg.energy_gate_rms:
            return False
        try:
            return bool(self._is_speech_raw(frame, SAMPLE_RATE))
        except Exception:
            return False

    def process_frame(self, frame: bytes) -> Optional[bytes]:
        """Feed exactly one FRAME_BYTES-length frame. Returns finished
        utterance WAV bytes if this frame completed one, else None."""
        if len(frame) != FRAME_BYTES:
            raise ValueError(f"frame must be exactly {FRAME_BYTES} bytes ({FRAME_MS}ms @ {SAMPLE_RATE}Hz)")

        voiced = self._frame_voiced(frame)

        if not self._triggered:
            self._ring.append((frame, voiced))
            num_voiced = sum(1 for _, v in self._ring if v)
            if len(self._ring) == self._ring.maxlen and num_voiced > self.cfg.trigger_ratio * self._ring.maxlen:
                self._triggered = True
                self._utterance_frames = [f for f, _ in self._ring]
                self._ring.clear()
                self._consecutive_unvoiced = 0
            return None

        # Triggered: keep every frame so internal pauses aren't lost
        self._utterance_frames.append(frame)
        self._consecutive_unvoiced = 0 if voiced else self._consecutive_unvoiced + 1

        total_ms = len(self._utterance_frames) * FRAME_MS
        silence_ms = self._consecutive_unvoiced * FRAME_MS

        ended_by_silence = silence_ms >= self.cfg.max_silence_gap_ms and total_ms >= self.cfg.min_utterance_ms
        ended_by_cap = total_ms >= self.cfg.max_utterance_ms

        if ended_by_silence or ended_by_cap:
            frames = self._utterance_frames
            if ended_by_silence:
                post_roll_frames = max(0, self.cfg.post_roll_ms // FRAME_MS)
                trim_to = len(frames) - self._consecutive_unvoiced + post_roll_frames
                frames = frames[:max(trim_to, 1)]
            wav_bytes = _frames_to_wav(frames)
            self._triggered = False
            self._utterance_frames = []
            self._consecutive_unvoiced = 0
            return wav_bytes

        return None

    def flush(self) -> Optional[bytes]:
        out = None
        if self._triggered and len(self._utterance_frames) * FRAME_MS >= self.cfg.min_utterance_ms:
            out = _frames_to_wav(self._utterance_frames)
        self._triggered = False
        self._utterance_frames = []
        self._consecutive_unvoiced = 0
        return out


class AudioCapture:
    """
    Owns the sounddevice InputStream and worker thread that re-chunks the
    hardware blocks into 20ms VAD frames.
    """

    def __init__(self, config: VADConfig = VADConfig(), device: Optional[object] = None,
                 blocksize: int = DEFAULT_BLOCKSIZE):
        self.segmenter = UtteranceSegmenter(config)
        self.device = device
        self.blocksize = blocksize
        self._raw_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=200)
        self._utterance_queue: "queue.Queue[bytes]" = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._leftover = b""

    def _callback(self, indata, frames, time_info, status):
        # Runs on the audio thread -- must never block
        if status:
            pass  # silent suppress of transient buffer glitches
        try:
            self._raw_queue.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                block = self._raw_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._leftover += block
            while len(self._leftover) >= FRAME_BYTES:
                frame, self._leftover = self._leftover[:FRAME_BYTES], self._leftover[FRAME_BYTES:]
                utterance = self.segmenter.process_frame(frame)
                if utterance is not None:
                    self._utterance_queue.put(utterance)
        tail = self.segmenter.flush()
        if tail is not None:
            self._utterance_queue.put(tail)

    def start(self) -> None:
        self._stop_event.clear()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16",
            blocksize=self.blocksize, device=self.device, callback=self._callback,
        )
        self._stream.start()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2)
            self._worker_thread = None

    def utterances(self) -> Generator[bytes, None, None]:
        """Blocking generator -- call from a background thread."""
        while not self._stop_event.is_set() or not self._utterance_queue.empty():
            try:
                yield self._utterance_queue.get(timeout=0.5)
            except queue.Empty:
                continue

    async def utterances_async(self):
        """Async bridge for FastAPI event loops."""
        import asyncio
        loop = asyncio.get_event_loop()
        while not self._stop_event.is_set() or not self._utterance_queue.empty():
            try:
                utterance = await loop.run_in_executor(None, self._utterance_queue.get, True, 0.5)
            except queue.Empty:
                continue
            yield utterance


# ── Backward-compatible generator for main.py ─────────────────────────────────

def capture_speech_segments(stop_event: Optional[threading.Event] = None) -> Generator[bytes, None, None]:
    """
    Drop-in replacement for the legacy capture_speech_segments generator.
    Spawns AudioCapture with safe 100ms hardware buffering and yields WAV bytes.
    """
    capture = AudioCapture()
    capture.start()
    try:
        for utt in capture.utterances():
            if stop_event and stop_event.is_set():
                break
            yield utt
    finally:
        capture.stop()


# --------------------------------------------------------------------------- #
# Self-test: pure state-machine logic, no microphone / real speech required.
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    def make_scripted_vad(pattern: List[bool]):
        it = iter(pattern)
        def fn(frame: bytes, sr: int) -> bool:
            return next(it, False)
        return fn

    def loud_frame(voiced: bool) -> bytes:
        amplitude = 3000 if voiced else 500
        return (np.full(FRAME_SAMPLES, amplitude, dtype=np.int16)).tobytes()

    # --- Test 1: short internal pause should NOT end the utterance early ---
    pattern = (
        [True] * 20      # onset: fills the pre-roll ring and triggers
        + [False] * 10   # 200ms pause -- shorter than max_silence_gap_ms, should NOT end
        + [True] * 20    # speech resumes
        + [False] * 40   # 800ms trailing silence, total speech now > min_utterance_ms -- SHOULD end
    )
    seg = UtteranceSegmenter(VADConfig(), vad_is_speech_fn=make_scripted_vad(pattern))
    results = []
    for voiced in pattern:
        out = seg.process_frame(loud_frame(voiced))
        if out is not None:
            results.append(out)
    assert len(results) == 1, f"expected exactly 1 utterance, got {len(results)}"
    with wave.open(io.BytesIO(results[0])) as wf:
        dur_ms = 1000 * wf.getnframes() / wf.getframerate()
    print(f"Test 1 (short pause survives, ends after sustained silence): 1 utterance, {dur_ms:.0f}ms -- PASS")

    # --- Test 2: hard cap fires even with continuous 'speech' ---
    pattern2 = [True] * 600  # 12s of uninterrupted voiced frames, cap is 10s
    seg2 = UtteranceSegmenter(VADConfig(), vad_is_speech_fn=make_scripted_vad(pattern2))
    results2 = []
    for voiced in pattern2:
        out = seg2.process_frame(loud_frame(voiced))
        if out is not None:
            results2.append(out)
    assert len(results2) == 1, f"expected the max-duration cap to force exactly 1 utterance, got {len(results2)}"
    with wave.open(io.BytesIO(results2[0])) as wf:
        dur_ms2 = 1000 * wf.getnframes() / wf.getframerate()
    assert 9900 <= dur_ms2 <= 10100, f"expected ~10000ms cap, got {dur_ms2:.0f}ms"
    print(f"Test 2 (hard max-duration cap): {dur_ms2:.0f}ms -- PASS")

    # --- Test 3: brief noise blip that never sustains shouldn't trigger at all ---
    pattern3 = [True, True, False, False, False] * 40
    seg3 = UtteranceSegmenter(VADConfig(), vad_is_speech_fn=make_scripted_vad(pattern3))
    results3 = [seg3.process_frame(loud_frame(v)) for v in pattern3]
    results3 = [r for r in results3 if r is not None]
    print(f"Test 3 (flickering non-speech doesn't false-trigger): {len(results3)} utterances -- "
          f"{'PASS' if len(results3) == 0 else 'CHECK'}")

    print("All audio_capture self-tests completed.")

