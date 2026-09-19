"""
EmpathAI — Live Face Mapping & Calibrated Emotion HUD Viewer
[OPTIONAL DEBUG EXTRA — Outside default PROMPT.md pipeline]

This script is an optional developer HUD and debugging tool for inspecting
MediaPipe face landmarks, testing FACS blendshapes, and interactive resting-face calibration.
It is NOT part of the default runtime or inference pipeline.

Features:
  • Real-time webcam feed (30 FPS)
  • MediaPipe 468-point 3D Face Mesh overlay
  • Tight square face crop for ViT
  • FACS Geometric Analysis + ViT Fusion
  • Per-user Neutral Face Calibration (press [C])
  • Hysteresis + EMA Temporal Smoothing (zero flickering)
  • 7-Emotion Live Spectrum Panel
  • Live audio vocal tone & speech transcript HUD banner
  • Controls:
      [C] Calibrate Neutral Baseline (hold natural resting face for 2s)
      [M] Toggle Face Mesh wireframe
      [B] Toggle Bounding Box
      [S] Toggle 7-Emotion Spectrum panel
      [Q] / [ESC] Quit
"""

import os
import sys
import time
import json
import threading
import queue
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarksConnections

from face_emotion import (
    EmotionRecognizer,
    crop_face_from_landmarks,
    vit_pipeline_output_to_scores,
    _get_face_emotion_pipeline,
    EMOTIONS,
)
from PIL import Image

# HUD Palette (BGR)
COLOR_BG_HUD       = (16, 18, 22)
COLOR_TEXT_WHITE   = (240, 242, 248)
COLOR_TEXT_DIM     = (150, 155, 168)
COLOR_CYAN         = (240, 200, 0)     # Neon Cyan
COLOR_GREEN        = (60, 220, 80)     # Neon Green
COLOR_YELLOW       = (0, 215, 255)     # Amber Yellow
COLOR_RED          = (60, 60, 245)     # Alert Red
COLOR_PURPLE       = (220, 70, 180)    # Soft Purple
COLOR_ORANGE       = (30, 140, 255)    # Vibrant Orange
COLOR_MESH_LINE    = (130, 180, 50)    # Subtle turquoise mesh lines
COLOR_CONTOUR_LINE = (230, 200, 30)    # Distinct feature contour

EMOTION_COLORS = {
    "happy":    COLOR_GREEN,
    "neutral":  (200, 180, 100),
    "surprise": COLOR_YELLOW,
    "sad":      COLOR_PURPLE,
    "angry":    COLOR_RED,
    "disgust":  COLOR_ORANGE,
    "fear":     (190, 50, 210),
}


class LiveFaceViewer:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.show_mesh = True
        self.show_box = True
        self.show_spectrum = True

        # Emotion Recognizer instance
        cal_path = Path(__file__).parent / "calibration_profile.json"
        self.recognizer = EmotionRecognizer(calibration_path=cal_path, calibration_frames=60)

        # Real-time state
        self.current_label = "neutral"
        self.current_confidence = 0.0
        self.scores = {e: 0.0 for e in EMOTIONS}
        self.geometric_scores = {}
        self.speech_emotion = "listening..."
        self.transcript_text = ""
        self.fps = 0.0
        self.status_msg = "Press [C] to calibrate resting face"
        self.status_time = time.time()

        # Queue for async ViT inference
        self.vit_queue = queue.Queue(maxsize=1)
        self.latest_vit_scores = {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}
        self.running = True

        # Initialize MediaPipe Face Landmarker
        print("[INFO] Loading MediaPipe Face Landmarker...", flush=True)
        model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        print("[OK] Face Landmarker initialized.", flush=True)

    def notify(self, text):
        self.status_msg = text
        self.status_time = time.time()

    def start_vit_worker(self):
        """Runs ViT inference asynchronously so video rendering stays at 30+ FPS."""
        def _worker():
            try:
                pipe = _get_face_emotion_pipeline()
                print("[OK] ViT facial emotion model loaded.", flush=True)
            except Exception as e:
                print(f"[WARN] ViT loader notice: {e}", flush=True)
                return

            while self.running:
                try:
                    crop_rgb = self.vit_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                try:
                    pil_img = Image.fromarray(crop_rgb)
                    raw_results = pipe(pil_img, top_k=None)
                    self.latest_vit_scores = vit_pipeline_output_to_scores(raw_results)
                except Exception:
                    pass

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def start_audio_listener(self):
        """Listens to the backend websocket stream for audio emotion and transcript."""
        def _audio_ws():
            import asyncio
            import websockets

            async def _listen():
                # camera=-1 tells the backend not to touch or lock the webcam hardware
                url = "ws://localhost:8000/stream?camera=-1"
                while self.running:
                    try:
                        async with websockets.connect(url, ping_interval=None) as ws:
                            while self.running:
                                msg = await ws.recv()
                                data = json.loads(msg)
                                if data.get("type") == "audio":
                                    speech = data.get("speech", {})
                                    tx = data.get("transcript", {})
                                    if "emotion" in speech:
                                        self.speech_emotion = speech["emotion"]
                                    if "transcript" in tx and tx["transcript"]:
                                        self.transcript_text = tx["transcript"]
                    except Exception:
                        await asyncio.sleep(2.0)

            try:
                asyncio.run(_listen())
            except Exception:
                pass

        t = threading.Thread(target=_audio_ws, daemon=True)
        t.start()

    def draw_face_mesh(self, frame, landmarks):
        h, w, _ = frame.shape
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        overlay = frame.copy()
        for conn in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
            p1 = pts[conn.start]
            p2 = pts[conn.end]
            cv2.line(overlay, p1, p2, COLOR_MESH_LINE, 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        for conn in FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS:
            p1 = pts[conn.start]
            p2 = pts[conn.end]
            cv2.line(frame, p1, p2, COLOR_CONTOUR_LINE, 1, cv2.LINE_AA)

        for conn in FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS:
            p1 = pts[conn.start]
            p2 = pts[conn.end]
            cv2.line(frame, p1, p2, COLOR_CYAN, 2, cv2.LINE_AA)

        for conn in FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS:
            p1 = pts[conn.start]
            p2 = pts[conn.end]
            cv2.line(frame, p1, p2, COLOR_CYAN, 2, cv2.LINE_AA)

    def draw_spectrum_panel(self, frame):
        if not self.show_spectrum:
            return

        h, w, _ = frame.shape
        panel_w = 210
        panel_x = w - panel_w - 15
        panel_y = 85
        panel_h = 240

        card = frame[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w].copy()
        cv2.rectangle(card, (0, 0), (panel_w, panel_h), (14, 16, 20), -1)
        cv2.addWeighted(card, 0.82, frame[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w], 0.18, 0, frame[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w])
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (50, 58, 70), 1)

        cv2.putText(frame, "EMOTION SPECTRUM", (panel_x + 15, panel_y + 22), cv2.FONT_HERSHEY_DUPLEX, 0.45, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        bar_start_y = panel_y + 44
        bar_max_w = 95

        for i, emo in enumerate(EMOTIONS):
            prob = self.scores.get(emo, 0.0)
            col = EMOTION_COLORS.get(emo, COLOR_CYAN)
            y = bar_start_y + i * 26

            is_curr = (emo == self.current_label)
            cv2.putText(frame, emo[:3].upper(), (panel_x + 12, y + 11), cv2.FONT_HERSHEY_DUPLEX, 0.42, COLOR_TEXT_WHITE if is_curr else COLOR_TEXT_DIM, 1, cv2.LINE_AA)

            meter_x = panel_x + 52
            fill_w = int(bar_max_w * max(0.0, min(1.0, prob)))
            cv2.rectangle(frame, (meter_x, y), (meter_x + bar_max_w, y + 13), (35, 38, 48), -1)
            if fill_w > 0:
                cv2.rectangle(frame, (meter_x, y), (meter_x + fill_w, y + 13), col, -1)
            cv2.rectangle(frame, (meter_x, y), (meter_x + bar_max_w, y + 13), (70, 78, 90), 1)

            cv2.putText(frame, f"{int(prob * 100)}%", (meter_x + bar_max_w + 8, y + 11), cv2.FONT_HERSHEY_DUPLEX, 0.4, col if prob > 0.3 else COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        # Calibration status badge
        status_text = "CALIBRATED [OK]" if self.recognizer.is_calibrated else "NOT CALIBRATED [C]"
        status_col = COLOR_GREEN if self.recognizer.is_calibrated else COLOR_YELLOW
        cv2.putText(frame, status_text, (panel_x + 12, panel_y + panel_h - 10), cv2.FONT_HERSHEY_DUPLEX, 0.38, status_col, 1, cv2.LINE_AA)

    def draw_calibration_modal(self, frame, progress):
        """Overlay calibration progress bar when user holds natural resting face."""
        h, w, _ = frame.shape
        modal_w, modal_h = 440, 110
        x0 = (w - modal_w) // 2
        y0 = (h - modal_h) // 2

        overlay = frame[y0:y0 + modal_h, x0:x0 + modal_w].copy()
        cv2.rectangle(overlay, (0, 0), (modal_w, modal_h), (18, 20, 26), -1)
        cv2.addWeighted(overlay, 0.92, frame[y0:y0 + modal_h, x0:x0 + modal_w], 0.08, 0, frame[y0:y0 + modal_h, x0:x0 + modal_w])
        cv2.rectangle(frame, (x0, y0), (x0 + modal_w, y0 + modal_h), COLOR_YELLOW, 2)

        cv2.putText(frame, "CALIBRATING NEUTRAL BASELINE", (x0 + 20, y0 + 32), cv2.FONT_HERSHEY_DUPLEX, 0.58, COLOR_YELLOW, 1, cv2.LINE_AA)
        cv2.putText(frame, "Hold a natural, relaxed resting face...", (x0 + 20, y0 + 56), cv2.FONT_HERSHEY_DUPLEX, 0.44, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

        # Progress bar
        bar_x = x0 + 20
        bar_y = y0 + 72
        bar_w = modal_w - 40
        bar_h = 16
        fill_w = int(bar_w * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (35, 40, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), COLOR_GREEN, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 110, 130), 1)
        cv2.putText(frame, f"{int(progress * 100)}%", (bar_x + bar_w - 40, bar_y + 12), cv2.FONT_HERSHEY_DUPLEX, 0.42, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

    def draw_hud(self, frame, bbox):
        h, w, _ = frame.shape
        emo_color = EMOTION_COLORS.get(self.current_label.lower(), COLOR_CYAN)

        # ── Face Bounding Box with Tech Brackets ──────────────────
        if self.show_box and bbox is not None:
            x1, y1, x2, y2 = bbox
            bw = int((x2 - x1) * 0.2)
            bh = int((y2 - y1) * 0.2)
            t = 2

            cv2.line(frame, (x1, y1), (x1 + bw, y1), emo_color, t)
            cv2.line(frame, (x1, y1), (x1, y1 + bh), emo_color, t)
            cv2.line(frame, (x2, y1), (x2 - bw, y1), emo_color, t)
            cv2.line(frame, (x2, y1), (x2, y1 + bh), emo_color, t)
            cv2.line(frame, (x1, y2), (x1 + bw, y2), emo_color, t)
            cv2.line(frame, (x1, y2), (x1, y2 - bh), emo_color, t)
            cv2.line(frame, (x2, y2), (x2 - bw, y2), emo_color, t)
            cv2.line(frame, (x2, y2), (x2 - bw, y2), emo_color, t)

            # Floating emotion badge
            badge_text = f"{self.current_label.upper()} {int(self.current_confidence * 100)}%"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
            badge_y = max(35, y1 - 12)
            cv2.rectangle(frame, (x1, badge_y - th - 8), (x1 + tw + 16, badge_y + 4), (18, 20, 24), -1)
            cv2.rectangle(frame, (x1, badge_y - th - 8), (x1 + tw + 16, badge_y + 4), emo_color, 1)
            cv2.putText(frame, badge_text, (x1 + 8, badge_y - 2), cv2.FONT_HERSHEY_DUPLEX, 0.65, emo_color, 1, cv2.LINE_AA)

        # ── Top HUD Header ───────────────────────────────────────
        hud_bar_h = 75
        hud_overlay = frame[0:hud_bar_h, 0:w].copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_bar_h), COLOR_BG_HUD, -1)
        cv2.addWeighted(hud_overlay, 0.88, frame[0:hud_bar_h, 0:w], 0.12, 0, frame[0:hud_bar_h, 0:w])
        cv2.line(frame, (0, hud_bar_h), (w, hud_bar_h), (48, 56, 68), 1)

        cv2.putText(frame, "EMPATHAI  |  FACS-CALIBRATED EMOTION HUD", (20, 26), cv2.FONT_HERSHEY_DUPLEX, 0.55, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w - 110, 26), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        # Emotion Display
        cv2.putText(frame, "FACIAL EXPRESSION:", (20, 56), cv2.FONT_HERSHEY_DUPLEX, 0.55, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, self.current_label.upper(), (215, 56), cv2.FONT_HERSHEY_DUPLEX, 0.72, emo_color, 2, cv2.LINE_AA)

        # Confidence Bar
        bar_x = 365
        bar_w = 160
        bar_h = 13
        bar_y = 45
        fill_w = int(bar_w * max(0.0, min(1.0, self.current_confidence)))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (36, 42, 52), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), emo_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (90, 100, 115), 1)
        cv2.putText(frame, f"{int(self.current_confidence * 100)}%", (bar_x + bar_w + 10, 56), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

        # Notification
        if time.time() - self.status_time < 3.5:
            cv2.putText(frame, f">> {self.status_msg}", (bar_x + bar_w + 65, 56), cv2.FONT_HERSHEY_DUPLEX, 0.45, COLOR_YELLOW, 1, cv2.LINE_AA)

        # ── Draw 7-Emotion Spectrum Panel ────────────────────────
        self.draw_spectrum_panel(frame)

        # ── Bottom Audio / Transcript Card ────────────────────────
        bot_h = 55
        bot_overlay = frame[h - bot_h:h, 0:w].copy()
        cv2.rectangle(bot_overlay, (0, 0), (w, bot_h), COLOR_BG_HUD, -1)
        cv2.addWeighted(bot_overlay, 0.88, frame[h - bot_h:h, 0:w], 0.12, 0, frame[h - bot_h:h, 0:w])
        cv2.line(frame, (0, h - bot_h), (w, h - bot_h), (48, 56, 68), 1)

        cv2.putText(frame, f"MIC TONE: {self.speech_emotion.upper()}", (20, h - 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLOR_CYAN, 1, cv2.LINE_AA)
        tx = self.transcript_text if self.transcript_text else "(listening for speech...)"
        if len(tx) > 42:
            tx = tx[:39] + "..."
        cv2.putText(frame, f'SPEECH: "{tx}"', (240, h - 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

        hint = "[C] Calibrate (2s)  [R] Reset  [M] Mesh  [B] Box  [S] Spectrum  [Q] Exit"
        cv2.putText(frame, hint, (20, h - 10), cv2.FONT_HERSHEY_DUPLEX, 0.38, (140, 148, 160), 1, cv2.LINE_AA)

    def run(self):
        print("\n" + "=" * 65, flush=True)
        print("  EMPATHAI — REAL-TIME FACE MAPPING & FACS CALIBRATION HUD", flush=True)
        print("=" * 65, flush=True)
        print("  • Controls:", flush=True)
        print("      [C] Calibrate resting face baseline (hold neutral for ~2s)", flush=True)
        print("      [R] Reset baseline to defaults (when switching users)", flush=True)
        print("      [M] Toggle Face Mesh wireframe", flush=True)
        print("      [B] Toggle Bounding Box & Badge", flush=True)
        print("      [S] Toggle 7-Emotion Spectrum panel", flush=True)
        print("      [Q] or [ESC] to Exit\n", flush=True)

        self.start_vit_worker()
        self.start_audio_listener()

        print(f"[INFO] Connecting to camera index {self.camera_index}...", flush=True)
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[WARN] DirectShow open failed, trying default camera backend...", flush=True)
            cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera index {self.camera_index}.", flush=True)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("[OK] Camera active. Launching display window...", flush=True)

        window_name = "EmpathAI — Live Face Mapping & FACS Calibration HUD"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1024, 680)

        prev_time = time.time()
        last_vit_time = 0

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.02)
                    continue

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                now = time.time()
                dt = now - prev_time
                prev_time = now
                if dt > 0:
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

                # Run MediaPipe Face Landmarker
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = self.landmarker.detect(mp_image)

                bbox = None
                calibrating_progress = None

                if res.face_landmarks and len(res.face_landmarks) > 0:
                    landmarks = res.face_landmarks[0]
                    blendshapes = res.face_blendshapes[0] if res.face_blendshapes else None

                    # Tight square crop for ViT
                    crop_bgr, bbox = crop_face_from_landmarks(frame, landmarks)

                    # Send to async ViT worker every 0.3s
                    if now - last_vit_time > 0.30:
                        if self.vit_queue.empty() and crop_bgr.size > 0:
                            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                            self.vit_queue.put(crop_rgb)
                            last_vit_time = now

                    # Process through Emotion Recognizer
                    em_res = self.recognizer.process_frame(landmarks, blendshapes, self.latest_vit_scores)

                    if em_res.label == "calibrating":
                        calibrating_progress = em_res.calibration_progress
                    else:
                        self.current_label = em_res.label
                        self.current_confidence = em_res.confidence
                        self.scores = em_res.scores
                        self.geometric_scores = em_res.geometric_scores

                    if self.show_mesh:
                        self.draw_face_mesh(frame, landmarks)

                # Draw Top, Bottom, and Spectrum HUD overlays
                self.draw_hud(frame, bbox)

                # If calibrating, draw the progress modal
                if calibrating_progress is not None:
                    self.draw_calibration_modal(frame, calibrating_progress)

                # Show frame
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key in (ord('c'), ord('C')):
                    self.recognizer.start_calibration()
                    self.notify("Calibrating neutral baseline...")
                elif key in (ord('r'), ord('R')):
                    self.recognizer.reset_calibration()
                    self.notify("Baseline reset to defaults [OK]")
                elif key in (ord('s'), ord('S')):
                    self.show_spectrum = not self.show_spectrum
                elif key in (ord('m'), ord('M')):
                    self.show_mesh = not self.show_mesh
                elif key in (ord('b'), ord('B')):
                    self.show_box = not self.show_box

        except (KeyboardInterrupt, SystemExit):
            print("\n[STOP] Interrupt received, closing...", flush=True)
        finally:
            self.running = False
            if 'cap' in locals() and cap is not None and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            if hasattr(self, "landmarker") and self.landmarker is not None:
                try:
                    self.landmarker.close()
                except Exception:
                    pass
            print("[INFO] Face viewer stopped cleanly.", flush=True)


if __name__ == "__main__":
    try:
        viewer = LiveFaceViewer(camera_index=0)
        viewer.run()
    except (KeyboardInterrupt, SystemExit):
        print("\n[STOP] Stopped by user.", flush=True)
