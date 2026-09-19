"""
Phase 1 — Calibrated, Jitter-Resistant Facial Emotion Pipeline for EmpathAI

Fuses:
  1. MediaPipe FaceLandmarker (478 3D landmarks + 52 FACS blendshapes)
  2. Scale-normalized facial geometric ratios (mouth corner angle, eye aperture, jaw drop)
  3. Hugging Face ViT model (`trpakov/vit-face-expression`) on tight square crops
  4. Per-user Neutral Calibration (learning resting face baseline to cancel out bias)
  5. Hysteresis + EMA Temporal Smoothing to prevent frame-to-frame label flicker
"""

from __future__ import annotations

import io
import os
import time
import json
import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import pipeline as hf_pipeline
try:
    from vram_logger import log_vram
except ImportError:
    def log_vram(label: str = ""):
        return None

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "trpakov/vit-face-expression"

EMOTIONS: List[str] = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

BLENDSHAPE_NAMES: List[str] = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft",
    "noseSneerRight",
]

# Landmark indices for geometric FACS cross-checks (MediaPipe 468/478 topology)
LM_LEFT_EYE_OUTER, LM_RIGHT_EYE_OUTER = 33, 263
LM_LEFT_EYE_UPPER, LM_LEFT_EYE_LOWER = 159, 145
LM_RIGHT_EYE_UPPER, LM_RIGHT_EYE_LOWER = 386, 374
LM_MOUTH_LEFT_CORNER, LM_MOUTH_RIGHT_CORNER = 61, 291
LM_UPPER_LIP_CENTER, LM_LOWER_LIP_CENTER = 13, 14
LM_CHIN, LM_NOSE_TIP = 152, 4


# ── ViT & MediaPipe Loaders ──────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_face_emotion_pipeline():
    """Load ViT face-expression classifier once, keep in memory."""
    logger.info("Loading face emotion model %s on %s …", MODEL_ID, DEVICE)
    pipe = hf_pipeline(
        "image-classification",
        model=MODEL_ID,
        device=0 if DEVICE == "cuda" else -1,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    logger.info("Face emotion model loaded.")
    log_vram("after ViT face model load")
    return pipe


_FACE_LANDMARKER = None

def _get_face_landmarker():
    global _FACE_LANDMARKER
    if _FACE_LANDMARKER is not None:
        return _FACE_LANDMARKER
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
    if os.path.exists(model_path):
        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                num_faces=1,
            )
            _FACE_LANDMARKER = vision.FaceLandmarker.create_from_options(options)
            logger.info("MediaPipe FaceLandmarker loaded from: %s", model_path)
            return _FACE_LANDMARKER
        except Exception as e:
            logger.warning("Failed to initialize MediaPipe FaceLandmarker: %s", e)
    return None


# ── Low-Level Geometric & FACS Helpers ────────────────────────────────────────

def blendshapes_to_dict(face_blendshapes) -> Dict[str, float]:
    """Convert MediaPipe's `res.face_blendshapes[0]` to a dict."""
    if not face_blendshapes:
        return {name: 0.0 for name in BLENDSHAPE_NAMES}
    out = {c.category_name: float(c.score) for c in face_blendshapes}
    for name in BLENDSHAPE_NAMES:
        out.setdefault(name, 0.0)
    return out


def _dist(a, b) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def compute_geometric_features(landmarks: Sequence) -> Dict[str, float]:
    """Scale-normalized geometric ratios that complement blendshapes."""
    if len(landmarks) < 468:
        return {
            "mouth_corner_angle": 0.0,
            "eye_aperture_left": 0.0,
            "eye_aperture_right": 0.0,
            "jaw_opening": 0.0,
        }

    face_scale = _dist(landmarks[LM_LEFT_EYE_OUTER], landmarks[LM_RIGHT_EYE_OUTER])
    face_scale = max(face_scale, 1e-4)

    mouth_center_y = (landmarks[LM_UPPER_LIP_CENTER].y + landmarks[LM_LOWER_LIP_CENTER].y) / 2.0
    mouth_corners_y = (landmarks[LM_MOUTH_LEFT_CORNER].y + landmarks[LM_MOUTH_RIGHT_CORNER].y) / 2.0
    mouth_corner_angle = (mouth_center_y - mouth_corners_y) / face_scale

    eye_aperture_left = _dist(landmarks[LM_LEFT_EYE_UPPER], landmarks[LM_LEFT_EYE_LOWER]) / face_scale
    eye_aperture_right = _dist(landmarks[LM_RIGHT_EYE_UPPER], landmarks[LM_RIGHT_EYE_LOWER]) / face_scale
    jaw_opening = _dist(landmarks[LM_UPPER_LIP_CENTER], landmarks[LM_LOWER_LIP_CENTER]) / face_scale

    return {
        "mouth_corner_angle": mouth_corner_angle,
        "eye_aperture_left": eye_aperture_left,
        "eye_aperture_right": eye_aperture_right,
        "jaw_opening": jaw_opening,
    }


def activate(raw: float, baseline: float, floor: float = 0.018, gain: float = 2.0) -> float:
    """Calibrated activation above user's resting baseline in [0, 1]."""
    delta = raw - baseline
    if delta < floor:
        return 0.0
    headroom = max(1.0 - baseline, 1e-3)
    return float(np.clip((delta / headroom) * gain, 0.0, 1.0))


def combine_cues(cues: Sequence[float], weights: Sequence[float], min_active: int = 2,
                 active_threshold: float = 0.15) -> float:
    """Multi-cue gated combination to stop single noisy spikes."""
    weights_arr = np.asarray(weights, dtype=float)
    cues_arr = np.asarray(cues, dtype=float)
    active = int(np.sum(cues_arr > active_threshold))
    weighted = float(np.sum(cues_arr * weights_arr) / np.sum(weights_arr))
    if active < min_active:
        weighted *= (active / max(min_active, 1))
    return float(np.clip(weighted, 0.0, 1.0))


def normalize(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in scores.values())
    if total <= 1e-6:
        n = len(scores)
        return {k: 1.0 / n for k in scores}
    return {k: max(v, 0.0) / total for k, v in scores.items()}


# ── Calibration Profile & Neutral Calibrator ─────────────────────────────────

# Maximum realistic baselines for a relaxed resting face.
# Prevents squinting/looking down at the monitor from corrupting angry/sad thresholds.
MAX_NEUTRAL_BASELINES: Dict[str, float] = {
    "browDownLeft": 0.18,
    "browDownRight": 0.18,
    "eyeSquintLeft": 0.22,
    "eyeSquintRight": 0.22,
    "mouthFrownLeft": 0.08,
    "mouthFrownRight": 0.08,
    "mouthPressLeft": 0.12,
    "mouthPressRight": 0.12,
    "mouthSmileLeft": 0.12,
    "mouthSmileRight": 0.12,
    "noseSneerLeft": 0.06,
    "noseSneerRight": 0.06,
}


@dataclass
class CalibrationProfile:
    blendshape_baseline: Dict[str, float] = field(default_factory=dict)
    geometric_baseline: Dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    created_at: float = 0.0

    def sanitize(self) -> None:
        """Clamp any outlier baselines that prevent negative emotions from firing."""
        for k, max_val in MAX_NEUTRAL_BASELINES.items():
            if k in self.blendshape_baseline and self.blendshape_baseline[k] > max_val:
                self.blendshape_baseline[k] = max_val

    @classmethod
    def default_profile(cls) -> "CalibrationProfile":
        blend = {name: 0.02 for name in BLENDSHAPE_NAMES}
        blend["browDownLeft"] = 0.04
        blend["browDownRight"] = 0.04
        blend["eyeSquintLeft"] = 0.06
        blend["eyeSquintRight"] = 0.06
        blend["mouthFrownLeft"] = 0.02
        blend["mouthFrownRight"] = 0.02
        geo = {
            "mouth_corner_angle": 0.0,
            "eye_aperture_left": 0.08,
            "eye_aperture_right": 0.08,
            "jaw_opening": 0.01,
        }
        return cls(blendshape_baseline=blend, geometric_baseline=geo, n_samples=60, created_at=time.time())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationProfile":
        data = json.loads(Path(path).read_text())
        return cls(**data)


class NeutralCalibrator:
    """Collects frames while the user holds a neutral, resting expression."""

    def __init__(self, n_frames: int = 60, jitter_std_warn: float = 0.08):
        self.n_frames = n_frames
        self.jitter_std_warn = jitter_std_warn
        self._blend_samples: List[Dict[str, float]] = []
        self._geo_samples: List[Dict[str, float]] = []

    @property
    def is_complete(self) -> bool:
        return len(self._blend_samples) >= self.n_frames

    @property
    def progress(self) -> float:
        return min(1.0, len(self._blend_samples) / self.n_frames)

    def add_frame(self, blendshapes: Dict[str, float], geometric: Dict[str, float]) -> None:
        if self.is_complete:
            return
        self._blend_samples.append(blendshapes)
        self._geo_samples.append(geometric)

    def finalize(self) -> Tuple[CalibrationProfile, List[str]]:
        warnings: List[str] = []
        blend_keys = self._blend_samples[0].keys() if self._blend_samples else []
        geo_keys = self._geo_samples[0].keys() if self._geo_samples else []

        blend_baseline = {}
        for k in blend_keys:
            vals = np.array([s.get(k, 0.0) for s in self._blend_samples])
            raw_mean = float(vals.mean())
            if k in MAX_NEUTRAL_BASELINES and raw_mean > MAX_NEUTRAL_BASELINES[k]:
                warnings.append(f"baseline '{k}' clamped from {raw_mean:.3f} to {MAX_NEUTRAL_BASELINES[k]:.3f}")
                raw_mean = MAX_NEUTRAL_BASELINES[k]
            blend_baseline[k] = raw_mean
            if vals.std() > self.jitter_std_warn:
                warnings.append(f"blendshape '{k}' unstable during calibration (std={vals.std():.3f})")

        geo_baseline = {}
        for k in geo_keys:
            vals = np.array([s.get(k, 0.0) for s in self._geo_samples])
            geo_baseline[k] = float(vals.mean())

        profile = CalibrationProfile(
            blendshape_baseline=blend_baseline,
            geometric_baseline=geo_baseline,
            n_samples=len(self._blend_samples),
            created_at=time.time(),
        )
        return profile, warnings


# ── FACS Heuristic Scoring ───────────────────────────────────────────────────

def facs_scores(calibrated_blend: Dict[str, float], geo_delta: Dict[str, float]) -> Dict[str, float]:
    b = calibrated_blend
    g = geo_delta

    def bget(name: str) -> float:
        return b.get(name, 0.0)

    scores: Dict[str, float] = {}

    # ANGRY: Corrugator brow lower is primary driver. Squint, mouth press, and nose sneer boost.
    brow_lower = max(bget("browDownLeft"), bget("browDownRight"))
    eye_squint = np.mean([bget("eyeSquintLeft"), bget("eyeSquintRight")])
    mouth_press = np.mean([bget("mouthPressLeft"), bget("mouthPressRight")])
    nose_sneer = max(bget("noseSneerLeft"), bget("noseSneerRight"))
    scores["angry"] = float(np.clip(
        brow_lower * 1.35 + 0.35 * eye_squint + 0.25 * mouth_press + 0.35 * nose_sneer,
        0.0, 1.0,
    ))

    # FEAR vs SURPRISE:
    eye_widen = max(bget("eyeWideLeft"), bget("eyeWideRight"))
    brow_raise = max(bget("browInnerUp"), np.mean([bget("browOuterUpLeft"), bget("browOuterUpRight")]))
    mouth_stretch = max(bget("mouthStretchLeft"), bget("mouthStretchRight"))
    jaw_drop = bget("jawOpen")

    # FEAR: Wide staring eyes + raised/tensed brows, without an open dropped jaw
    fear_cue = 0.75 * eye_widen + 0.40 * brow_raise + 0.30 * mouth_stretch
    # Suppress fear if jaw is gaping open (which is surprise)
    scores["fear"] = float(np.clip(fear_cue - 0.35 * max(0.0, jaw_drop - 0.15), 0.0, 1.0))

    # SURPRISE: Open dropped jaw + raised arched brows
    scores["surprise"] = float(np.clip(
        0.55 * jaw_drop + 0.35 * brow_raise + 0.25 * eye_widen,
        0.0, 1.0,
    ))

    # SAD: Lip-corner depression + inner-brow raise + downward geometric mouth angle
    lip_depress = max(bget("mouthFrownLeft"), bget("mouthFrownRight"))
    inner_brow = bget("browInnerUp")
    frown_geo = float(np.clip(-g.get("mouth_corner_angle", 0.0) * 6.0, 0.0, 1.0))
    scores["sad"] = float(np.clip(
        0.60 * max(lip_depress, frown_geo) + 0.45 * inner_brow,
        0.0, 1.0,
    ))

    # DISGUST: Nose sneer + upper-lip raise
    lip_raise = max(bget("mouthUpperUpLeft"), bget("mouthUpperUpRight"))
    scores["disgust"] = float(np.clip(0.70 * nose_sneer + 0.45 * lip_raise, 0.0, 1.0))

    # HAPPY: Smile + cheek raise (Duchenne marker) + upward mouth corner angle
    smile = max(bget("mouthSmileLeft"), bget("mouthSmileRight"))
    cheek_raise = np.mean([bget("cheekSquintLeft"), bget("cheekSquintRight")])
    smile_geo = float(np.clip(g.get("mouth_corner_angle", 0.0) * 4.5, 0.0, 1.0))
    scores["happy"] = float(np.clip(0.65 * max(smile, smile_geo) + 0.35 * cheek_raise, 0.0, 1.0))

    # NEUTRAL: Exponential decay as soon as any expression activates
    # Drops sharply from 1.0 -> 0.36 at 0.25 activation, allowing natural expressions to cleanly lead
    strongest_other = max(scores.values()) if scores else 0.0
    scores["neutral"] = float(np.clip(np.exp(-4.2 * strongest_other), 0.0, 1.0))

    for e in EMOTIONS:
        scores.setdefault(e, 0.0)
    return scores


# ── Fusion ───────────────────────────────────────────────────────────────────

@dataclass
class FusionConfig:
    # Balanced weights so ViT angry/sad predictions are not artificially suppressed
    vit_trust: Dict[str, float] = field(default_factory=lambda: {
        "angry": 0.88, "sad": 0.88, "disgust": 0.80, "fear": 0.80,
        "happy": 0.88, "neutral": 0.70, "surprise": 0.85,
    })
    geometric_boost: float = 1.6
    alpha_min: float = 0.20
    alpha_max: float = 0.85


def fuse(geometric_scores: Dict[str, float], vit_scores: Dict[str, float],
         cfg: FusionConfig = FusionConfig()) -> Dict[str, float]:
    g = normalize({e: geometric_scores.get(e, 0.0) for e in EMOTIONS})
    v_raw = {e: vit_scores.get(e, 0.0) for e in EMOTIONS}

    v_trusted = {e: v_raw[e] * cfg.vit_trust.get(e, 1.0) for e in EMOTIONS}
    v_adj = normalize(v_trusted)

    alpha = float(np.clip(max(g.values()) * cfg.geometric_boost, cfg.alpha_min, cfg.alpha_max))
    fused = {e: alpha * g[e] + (1.0 - alpha) * v_adj[e] for e in EMOTIONS}
    return normalize(fused)


# ── Temporal Smoothing ───────────────────────────────────────────────────────

class TemporalSmoother:
    """EMA smoothing + sliding-window hysteresis to prevent flickering."""

    def __init__(self, ema_alpha: float = 0.45, window: int = 5,
                 switch_margin: float = 0.04, min_hold_frames: int = 3):
        self.ema_alpha = ema_alpha
        self.window = window
        self.switch_margin = switch_margin
        self.min_hold_frames = min_hold_frames
        self.ema_scores: Dict[str, float] = {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}
        self._argmax_history: deque = deque(maxlen=window)
        self.current_label: str = "neutral"
        self._frames_since_switch: int = 0

    def reset(self) -> None:
        self.ema_scores = {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}
        self._argmax_history.clear()
        self.current_label = "neutral"
        self._frames_since_switch = 0

    def update(self, fused_scores: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        for e in EMOTIONS:
            self.ema_scores[e] = (self.ema_alpha * fused_scores.get(e, 0.0)
                                  + (1 - self.ema_alpha) * self.ema_scores[e])

        self._argmax_history.append(max(self.ema_scores, key=self.ema_scores.get))
        self._frames_since_switch += 1

        vote_counts: Dict[str, int] = {}
        for e in self._argmax_history:
            vote_counts[e] = vote_counts.get(e, 0) + 1
        candidate = max(vote_counts, key=vote_counts.get)

        if candidate != self.current_label:
            margin_ok = (self.ema_scores[candidate] - self.ema_scores[self.current_label]) > self.switch_margin
            hold_ok = self._frames_since_switch >= self.min_hold_frames
            if margin_ok and hold_ok:
                self.current_label = candidate
                self._frames_since_switch = 0

        return self.current_label, dict(self.ema_scores)


# ── Crop & ViT Format Helpers ────────────────────────────────────────────────

def crop_face_from_landmarks(frame_bgr: np.ndarray, landmarks: Sequence,
                             margin: float = 0.35) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Tight, margin-padded square crop matching FER2013 training distribution."""
    h, w = frame_bgr.shape[:2]
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    box_w, box_h = x_max - x_min, y_max - y_min
    side = max(box_w, box_h) * (1 + margin)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

    x0 = int(max(0, cx - side / 2))
    y0 = int(max(0, cy - side / 2))
    x1 = int(min(w, cx + side / 2))
    y1 = int(min(h, cy + side / 2))
    return frame_bgr[y0:y1, x0:x1], (x0, y0, x1, y1)


def vit_pipeline_output_to_scores(pipeline_output: List[dict]) -> Dict[str, float]:
    scores = {e: 0.0 for e in EMOTIONS}
    for item in pipeline_output:
        label = str(item["label"]).lower()
        if label in scores:
            scores[label] = float(item["score"])
    return normalize(scores)


# ── Emotion Recognizer Orchestrator ──────────────────────────────────────────

@dataclass
class EmotionResult:
    label: str
    confidence: float
    scores: Dict[str, float]
    geometric_scores: Dict[str, float]
    vit_scores: Dict[str, float]
    calibrated: bool
    calibration_progress: float


class EmotionRecognizer:
    def __init__(self, calibration_path: str | Path = None,
                 calibration_frames: int = 60,
                 fusion_cfg: Optional[FusionConfig] = None,
                 ema_alpha: float = 0.45, window: int = 5,
                 switch_margin: float = 0.04, min_hold_frames: int = 3):
        if calibration_path is None:
            calibration_path = Path(__file__).parent / "calibration_profile.json"
        self.calibration_path = Path(calibration_path)
        self.calibration_frames = calibration_frames
        self.fusion_cfg = fusion_cfg or FusionConfig()
        self.smoother = TemporalSmoother(ema_alpha, window, switch_margin, min_hold_frames)

        self.profile: Optional[CalibrationProfile] = None
        self.calibrator: Optional[NeutralCalibrator] = None

        if self.calibration_path.exists():
            try:
                self.profile = CalibrationProfile.load(self.calibration_path)
                self.profile.sanitize()
                self.profile.save(self.calibration_path)
            except Exception:
                self.profile = CalibrationProfile.default_profile()
                self.profile.save(self.calibration_path)
        else:
            self.profile = CalibrationProfile.default_profile()
            self.profile.save(self.calibration_path)

    @property
    def is_calibrated(self) -> bool:
        return self.profile is not None

    def start_calibration(self) -> None:
        self.calibrator = NeutralCalibrator(n_frames=self.calibration_frames)

    def cancel_calibration(self) -> None:
        self.calibrator = None

    def reset_calibration(self) -> None:
        """Reset calibration profile back to clean, balanced neutral baseline."""
        self.profile = CalibrationProfile.default_profile()
        self.profile.save(self.calibration_path)
        self.calibrator = None
        self.smoother.reset()

    def process_frame(self, landmarks: Sequence, face_blendshapes,
                      vit_scores: Dict[str, float]) -> EmotionResult:
        blend = blendshapes_to_dict(face_blendshapes)
        geo = compute_geometric_features(landmarks)

        if self.calibrator is not None and not self.calibrator.is_complete:
            self.calibrator.add_frame(blend, geo)
            progress = self.calibrator.progress
            if self.calibrator.is_complete:
                profile, warnings = self.calibrator.finalize()
                self.profile = profile
                self.profile.save(self.calibration_path)
                self.calibrator = None
                self.smoother.reset()
            return EmotionResult(
                label="calibrating", confidence=0.0,
                scores={e: 0.0 for e in EMOTIONS}, geometric_scores={}, vit_scores=vit_scores,
                calibrated=self.profile is not None, calibration_progress=progress,
            )

        if self.profile is None:
            v_adj = normalize({e: vit_scores.get(e, 0.0) * self.fusion_cfg.vit_trust.get(e, 1.0)
                               for e in EMOTIONS})
            label, smoothed = self.smoother.update(v_adj)
            return EmotionResult(
                label=label, confidence=round(smoothed[label], 4), scores=smoothed,
                geometric_scores={}, vit_scores=vit_scores,
                calibrated=False, calibration_progress=0.0,
            )

        calibrated_blend = {
            name: activate(blend.get(name, 0.0), self.profile.blendshape_baseline.get(name, 0.0))
            for name in BLENDSHAPE_NAMES
        }
        geo_delta = {k: v - self.profile.geometric_baseline.get(k, 0.0) for k, v in geo.items()}

        geometric_scores = normalize(facs_scores(calibrated_blend, geo_delta))
        fused = fuse(geometric_scores, vit_scores, self.fusion_cfg)
        label, smoothed = self.smoother.update(fused)

        return EmotionResult(
            label=label,
            confidence=round(smoothed[label], 4),
            scores={k: round(v, 4) for k, v in smoothed.items()},
            geometric_scores={k: round(v, 4) for k, v in geometric_scores.items()},
            vit_scores=vit_scores,
            calibrated=True,
            calibration_progress=1.0,
        )


# Global recognizer for API endpoints
_GLOBAL_RECOGNIZER = EmotionRecognizer()


# ── FastAPI / Core Prediction Entrypoint ─────────────────────────────────────

def predict_face_emotion(image_bytes: bytes, use_facs: Optional[bool] = None) -> dict:
    """
    Face emotion prediction pipeline.
    Default (PROMPT.md Phase 1):
      webcam frame in -> MediaPipe crop -> ViT model -> emotion label + confidence, returned as JSON.

    Optional extra:
      FACS blendshapes, neutral calibration baseline, and EMA temporal smoothing
      are gated behind `use_facs=True` (or env ENABLE_FACS=1), disabled by default.
    """
    if use_facs is None:
        use_facs = os.getenv("ENABLE_FACS", "false").lower() in ("true", "1", "yes")

    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image bytes.")

    landmarker = _get_face_landmarker()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    bbox = None
    landmarks_list = None
    blendshapes = None

    if landmarker is not None:
        try:
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect(mp_image)
            if res.face_landmarks and len(res.face_landmarks) > 0:
                lms = res.face_landmarks[0]
                crop_bgr, bbox = crop_face_from_landmarks(img_bgr, lms)
                face_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                if use_facs:
                    landmarks_list = [[round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)] for lm in lms]
                    blendshapes = res.face_blendshapes[0] if res.face_blendshapes else None
            else:
                face_rgb = rgb
        except Exception as e:
            logger.warning("MediaPipe error: %s", e)
            face_rgb = rgb
    else:
        face_rgb = rgb

    # Run ViT on tight face crop
    pil_image = Image.fromarray(face_rgb)
    pipe = _get_face_emotion_pipeline()
    raw_results = pipe(pil_image, top_k=None)
    vit_scores = vit_pipeline_output_to_scores(raw_results)

    # Optional extra: Process through Calibrated Emotion Recognizer (FACS + Calibration)
    if use_facs and landmarks_list and blendshapes:
        lms_obj = res.face_landmarks[0]
        result = _GLOBAL_RECOGNIZER.process_frame(lms_obj, blendshapes, vit_scores)
        top_label = result.label
        top_conf = result.confidence
        scores = result.scores
        geo_scores = result.geometric_scores
        out = {
            "emotion": top_label,
            "confidence": top_conf,
            "all_scores": scores,
            "vit_scores": vit_scores,
            "geometric_scores": geo_scores,
            "calibrated": _GLOBAL_RECOGNIZER.is_calibrated,
            "facs_enabled": True,
        }
        if landmarks_list is not None:
            out["landmarks"] = landmarks_list
    else:
        # Default spec: pure ViT on MediaPipe crop
        sorted_vit = sorted(vit_scores.items(), key=lambda x: x[1], reverse=True)
        top_label, top_conf = sorted_vit[0]
        out = {
            "emotion": top_label,
            "confidence": round(float(top_conf), 4),
            "all_scores": vit_scores,
            "vit_scores": vit_scores,
            "facs_enabled": False,
        }

    if bbox is not None:
        out["bbox"] = list(bbox)
    return out
