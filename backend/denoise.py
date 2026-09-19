"""
denoise.py - EmpathAI Audio Noise Suppression Layer
===================================================
Applies noise reduction to complete utterance WAVs (16kHz, mono, int16)
before vocal tone emotion recognition (speech_emotion.py) and transcription (transcript.py).

Three-tier backend selection tried at import time:
- Tier 1: Real RNNoise binding (rnnoise_wrapper / rnnoise) via 48kHz polyphase resampling.
- Tier 2: noisereduce library (direct 16kHz spectral gating).
- Tier 3: Built-in zero-dependency spectral gating via numpy & scipy.signal.stft / istft.

Safe by design: if denoising fails for any reason, it logs a warning and returns
the original un-denoised audio bytes unchanged without interrupting the pipeline.
"""
from __future__ import annotations

import io
import logging
import wave
from typing import Tuple

import numpy as np
import scipy.signal

logger = logging.getLogger("empathai.denoise")

SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2  # 16-bit
CHANNELS = 1


# ── WAV byte parsing & serialization helpers ──────────────────────────────────

def _wav_bytes_to_float32(wav_bytes: bytes) -> Tuple[np.ndarray, int, int]:
    """Parse 16-bit PCM WAV bytes to float32 samples in [-1.0, 1.0]."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if sampwidth != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got sample width {sampwidth} bytes")
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sr, channels


def _float32_to_wav_bytes(samples: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    """Convert float32 samples to 16kHz mono 16-bit PCM WAV bytes."""
    clamped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clamped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH_BYTES)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


# ── Tier 1: RNNoise (48kHz frame-based) ───────────────────────────────────────

def _init_rnnoise():
    try:
        import rnnoise_wrapper
        denoiser = rnnoise_wrapper.RNNoise()
        # Verify native library actually executes a test frame
        test_frame = np.zeros(480, dtype=np.int16)
        _ = denoiser.filter(test_frame)
        return denoiser
    except Exception:
        return None


def _denoise_rnnoise(samples: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    # Resample 16kHz -> 48kHz using polyphase filter (up=3, down=1)
    samples_48k = scipy.signal.resample_poly(samples, up=3, down=1)
    pcm_48k = np.clip(samples_48k * 32767.0, -32768, 32767).astype(np.int16)

    frame_len = 480  # 10ms at 48kHz
    pad_len = (frame_len - (len(pcm_48k) % frame_len)) % frame_len
    if pad_len > 0:
        pcm_48k = np.pad(pcm_48k, (0, pad_len))

    out_frames = []
    for i in range(0, len(pcm_48k), frame_len):
        chunk = pcm_48k[i:i + frame_len]
        out_chunk = _rnnoise.filter(chunk)
        out_frames.append(out_chunk)

    cleaned_48k = np.concatenate(out_frames).astype(np.float32) / 32768.0
    if pad_len > 0:
        cleaned_48k = cleaned_48k[:-pad_len]

    # Resample 48kHz -> 16kHz using polyphase filter (up=1, down=3)
    cleaned_16k = scipy.signal.resample_poly(cleaned_48k, up=1, down=3)
    if len(cleaned_16k) >= len(samples):
        return cleaned_16k[:len(samples)]
    return np.pad(cleaned_16k, (0, len(samples) - len(cleaned_16k)))


# ── Tier 2: noisereduce ────────────────────────────────────────────────────────

def _init_noisereduce():
    try:
        import noisereduce as nr
        return nr
    except Exception:
        return None


def _denoise_noisereduce(samples: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    return _noisereduce.reduce_noise(y=samples, sr=sr)


# ── Tier 3: Built-in Spectral Gating (scipy + numpy fallback) ─────────────────

def _denoise_spectral_gating(samples: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Zero-dependency spectral-gating fallback using only numpy and scipy.signal.stft/istft.
    - Estimates per-frequency-bin noise floor via 20th percentile magnitude across utterance.
    - Applies soft gain mask with a floor of 0.06 to avoid musical-noise artifacts.
    - Reconstructed via ISTFT.
    """
    if len(samples) < 512:
        return samples

    nperseg = 512
    noverlap = 384
    _, _, Zxx = scipy.signal.stft(samples, fs=sr, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Zxx)

    # 20th percentile across time frames (axis=-1)
    noise_floor = np.percentile(mag, 20, axis=-1, keepdims=True)

    # Soft gain mask with a floor of 0.06 to avoid musical-noise artifacts
    gain = np.clip(1.0 - (noise_floor / np.maximum(mag, 1e-8)), 0.06, 1.0)
    Zxx_clean = Zxx * gain

    _, clean = scipy.signal.istft(Zxx_clean, fs=sr, nperseg=nperseg, noverlap=noverlap)
    if len(clean) >= len(samples):
        return clean[:len(samples)]
    return np.pad(clean, (0, len(samples) - len(clean)))


# ── Backend Initialization (Tier 1 -> Tier 2 -> Tier 3) ──────────────────────

_rnnoise = _init_rnnoise()
if _rnnoise is not None:
    _ACTIVE_TIER = "Tier 1: rnnoise"
    _denoise_fn = _denoise_rnnoise
else:
    _noisereduce = _init_noisereduce()
    if _noisereduce is not None:
        _ACTIVE_TIER = "Tier 2: noisereduce"
        _denoise_fn = _denoise_noisereduce
    else:
        _ACTIVE_TIER = "Tier 3: spectral_gating"
        _denoise_fn = _denoise_spectral_gating

logger.info("denoise backend: %s", _ACTIVE_TIER)


# ── Primary Public Entrypoint ─────────────────────────────────────────────────

def _ensure_wav_bytes(raw_bytes: bytes) -> bytes:
    """Ensures audio is in 16kHz mono RIFF WAV container, converting WebM/MP3/OGG if needed."""
    if not raw_bytes or raw_bytes.startswith(b"RIFF"):
        return raw_bytes
    import os
    import subprocess
    import tempfile
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f_in:
            f_in.write(raw_bytes)
            in_path = f_in.name
        out_path = in_path + ".wav"
        try:
            cmd = [
                ffmpeg_exe, "-y", "-i", in_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
                out_path,
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
            if res.returncode == 0 and os.path.exists(out_path):
                with open(out_path, "rb") as wf:
                    return wf.read()
        finally:
            for p in (in_path, out_path):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
    except Exception as exc:
        logger.warning("Could not convert audio container to WAV: %s", exc)
    return raw_bytes


def denoise_wav_bytes(wav_bytes: bytes) -> bytes:
    """
    Denoises a complete utterance WAV (16kHz, mono, int16).
    Returns denoised WAV bytes in the same format.

    Wrapped in try/except: if any exception occurs, logs a warning and returns
    original un-denoised bytes unchanged so denoising never breaks an utterance.
    """
    if not wav_bytes:
        return wav_bytes
    try:
        wav_bytes = _ensure_wav_bytes(wav_bytes)
        samples, sr, _ = _wav_bytes_to_float32(wav_bytes)
        if len(samples) == 0:
            return wav_bytes

        cleaned_samples = _denoise_fn(samples, sr=sr)
        return _float32_to_wav_bytes(cleaned_samples, sr=sr)
    except Exception as exc:
        logger.warning("Audio denoising failed, falling back to raw audio: %s", exc)
        return wav_bytes


# ── Self-Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 1-second 440Hz tone mixed with white noise
    tone = 0.4 * np.sin(2 * np.pi * 440 * t)
    noise = np.random.normal(0, 0.15, len(t))
    noisy_samples = np.clip(tone + noise, -1.0, 1.0)

    input_wav = _float32_to_wav_bytes(noisy_samples, sr=sr)
    output_wav = denoise_wav_bytes(input_wav)

    out_samples, out_sr, channels = _wav_bytes_to_float32(output_wav)
    assert out_sr == sr, f"Expected sr={sr}, got {out_sr}"
    assert len(out_samples) == len(noisy_samples), f"Length mismatch: {len(out_samples)} vs {len(noisy_samples)}"

    rms_in = float(np.sqrt(np.mean(noisy_samples**2)))
    rms_out = float(np.sqrt(np.mean(out_samples**2)))

    assert rms_out < rms_in, f"Expected rms_out < rms_in, got {rms_out} >= {rms_in}"
    attenuation_pct = (1.0 - (rms_out / rms_in)) * 100.0

    print(f"denoise.py self-test ({_ACTIVE_TIER}): PASS -> RMS in: {rms_in:.4f}, RMS out: {rms_out:.4f} (attenuation: {attenuation_pct:.1f}%)")
