"""
speech_emotion.py
==================
Speech emotion recognition using Hugging Face
`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`, FP16 on CUDA.

Fixes applied:
1. FLAT PROBABILITIES ON QUIET AUDIO: normalize_audio() applies peak/RMS
   loudness normalization before feature extraction so quiet mic input has
   strong acoustic signal features.
2. CUDA DEVICE-SIDE ASSERTS: only input_values is cast to FP16; attention_mask
   remains in native int64/long dtype.
3. Softmax is computed in FP32 for numerical stability.
4. Provides both `SpeechEmotionRecognizer` and drop-in `predict_speech_emotion`.
"""
from __future__ import annotations

import io
import wave
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
try:
    from vram_logger import log_vram
except ImportError:
    def log_vram(label: str = ""):
        return None

logger = logging.getLogger(__name__)

MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
TARGET_SAMPLE_RATE = 16000


@dataclass
class NormalizationConfig:
    target_rms_dbfs: float = -20.0      # target loudness after normalization
    target_peak: float = 0.95           # peak amplitude ceiling
    max_gain_db: float = 25.0           # cap gain to prevent blowing up pure noise
    silence_rms_floor: float = 1e-4     # below this RMS, treat as silence


def _pcm16_wav_bytes_to_float32(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        sampwidth = wf.getsampwidth()
        n_channels = wf.getnchannels()
    if sampwidth != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got sample width {sampwidth} bytes")
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1)
    return pcm, sr


def normalize_audio(waveform: np.ndarray, cfg: NormalizationConfig = NormalizationConfig()) -> np.ndarray:
    """Peak/RMS loudness normalization with a noise-floor guard and a gain cap."""
    if waveform.size == 0:
        return waveform
    waveform = waveform - float(np.mean(waveform))  # remove DC offset
    rms = float(np.sqrt(np.mean(waveform ** 2)))
    if rms < cfg.silence_rms_floor:
        return waveform.astype(np.float32)

    target_rms = 10 ** (cfg.target_rms_dbfs / 20.0)
    gain = target_rms / rms
    max_gain = 10 ** (cfg.max_gain_db / 20.0)
    gain = min(gain, max_gain)

    out = waveform * gain
    peak = float(np.max(np.abs(out)))
    if peak > cfg.target_peak:
        out = out * (cfg.target_peak / peak)
    return out.astype(np.float32)


def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return waveform
    try:
        import torchaudio
        wav_t = torch.from_numpy(waveform).unsqueeze(0)
        resampled = torchaudio.functional.resample(wav_t, orig_sr, target_sr)
        return resampled.squeeze(0).numpy()
    except Exception:
        duration = waveform.shape[0] / orig_sr
        n_target = int(duration * target_sr)
        x_old = np.linspace(0, duration, num=waveform.shape[0])
        x_new = np.linspace(0, duration, num=n_target)
        return np.interp(x_new, x_old, waveform).astype(np.float32)


class SpeechEmotionRecognizer:
    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None,
                 use_fp16: Optional[bool] = None, norm_cfg: NormalizationConfig = NormalizationConfig()):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 if use_fp16 is not None else (self.device == "cuda")
        self.norm_cfg = norm_cfg

        logger.info("Loading speech emotion model %s on %s (fp16=%s)...", model_name, self.device, self.use_fp16)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(model_name)
        self.model.to(self.device)
        if self.use_fp16:
            self.model.half()
        self.model.eval()

        # Read labels from checkpoint configuration
        self.id2label: Dict[int, str] = dict(self.model.config.id2label)
        log_vram("after Wav2Vec2 speech model load")

    @torch.inference_mode()
    def predict(self, audio: Union[bytes, bytearray, np.ndarray], sample_rate: Optional[int] = None) -> Dict:
        """
        Predict emotion from audio bytes or numpy waveform.
        Returns: {"emotion": str, "confidence": float, "all_scores": dict}
        """
        if isinstance(audio, (bytes, bytearray)):
            waveform, sample_rate = _pcm16_wav_bytes_to_float32(bytes(audio))
        else:
            waveform = np.asarray(audio, dtype=np.float32)
            if sample_rate is None:
                raise ValueError("sample_rate is required when passing a raw numpy array")

        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = _resample(waveform, sample_rate, TARGET_SAMPLE_RATE)

        waveform = normalize_audio(waveform, self.norm_cfg)

        inputs = self.feature_extractor(
            waveform, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", padding=True,
        )
        input_values = inputs["input_values"].to(self.device)
        if self.use_fp16:
            input_values = input_values.half()

        model_kwargs = {"input_values": input_values}
        if "attention_mask" in inputs:
            model_kwargs["attention_mask"] = inputs["attention_mask"].to(self.device)

        logits = self.model(**model_kwargs).logits
        probs = torch.softmax(logits.float(), dim=-1).squeeze(0).cpu().numpy()

        all_scores = {self.id2label[i]: round(float(probs[i]), 4) for i in range(len(probs))}
        top_idx = int(np.argmax(probs))
        return {
            "emotion": self.id2label[top_idx],
            "confidence": round(float(probs[top_idx]), 4),
            "all_scores": all_scores,
        }


# ── Global singleton & drop-in prediction helper ──────────────────────────────

_GLOBAL_SPEECH_RECOGNIZER: Optional[SpeechEmotionRecognizer] = None

def predict_speech_emotion(audio_bytes: bytes) -> dict:
    """Drop-in functional entrypoint for main.py / FastAPI."""
    global _GLOBAL_SPEECH_RECOGNIZER
    if _GLOBAL_SPEECH_RECOGNIZER is None:
        _GLOBAL_SPEECH_RECOGNIZER = SpeechEmotionRecognizer()
    return _GLOBAL_SPEECH_RECOGNIZER.predict(audio_bytes)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Quiet "speech-shaped" signal: sine + noise at low amplitude.
    t = np.linspace(0, 1.0, TARGET_SAMPLE_RATE, endpoint=False)
    quiet = (0.01 * np.sin(2 * np.pi * 200 * t) + 0.002 * rng.standard_normal(t.shape)).astype(np.float32)
    normalized = normalize_audio(quiet)
    quiet_rms_db = 20 * np.log10(np.sqrt(np.mean(quiet ** 2)) + 1e-12)
    norm_rms_db = 20 * np.log10(np.sqrt(np.mean(normalized ** 2)) + 1e-12)
    print(f"Quiet input RMS: {quiet_rms_db:.1f} dBFS -> normalized RMS: {norm_rms_db:.1f} dBFS")
    assert norm_rms_db > quiet_rms_db + 10, "normalization should meaningfully raise a quiet signal's loudness"
    assert np.max(np.abs(normalized)) <= NormalizationConfig().target_peak + 1e-6, "must not exceed target_peak"

    # Near-silence should be left alone rather than amplified into fake "loud noise".
    silence = (1e-6 * rng.standard_normal(TARGET_SAMPLE_RATE)).astype(np.float32)
    norm_silence = normalize_audio(silence)
    assert np.allclose(norm_silence, silence - silence.mean(), atol=1e-5), "near-silence should not be gain-boosted"

    # Loud/clipping-risk input should be scaled down to target_peak, not left clipping.
    loud = (0.99 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    norm_loud = normalize_audio(loud)
    assert np.max(np.abs(norm_loud)) <= NormalizationConfig().target_peak + 1e-6

    # WAV round-trip decode check.
    pcm16 = (quiet * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())
    decoded, sr = _pcm16_wav_bytes_to_float32(buf.getvalue())
    assert sr == TARGET_SAMPLE_RATE
    assert np.allclose(decoded, pcm16.astype(np.float32) / 32768.0, atol=1e-6)

    print("All speech_emotion self-tests (normalization + WAV decode) passed.")
    print("Note: SpeechEmotionRecognizer.predict() itself requires downloading "
          f"'{MODEL_NAME}' from huggingface.co and a torch/transformers install "
          "to exercise end-to-end.")

