"""
Phase 1 — standalone test script for all three modality endpoints.
Run with: python test_phase1.py

Downloads a tiny sample image and audio clip from the internet
(or uses local files if provided) and hits each endpoint.
"""

import sys
import os
import requests

BASE_URL = "http://localhost:8000"


def _green(s): return f"\033[92m{s}\033[0m"
def _red(s):   return f"\033[91m{s}\033[0m"
def _bold(s):  return f"\033[1m{s}\033[0m"


def test_health():
    print(_bold("\n── /health ──────────────────────────────────"))
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    assert r.status_code == 200 and r.json()["status"] == "ok", f"FAIL: {r.text}"
    print(_green(f"  ✓ {r.json()}"))


def test_face_emotion(image_path: str):
    print(_bold(f"\n── /face-emotion  ({image_path}) ──────────────"))
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/face-emotion",
            files={"image": (os.path.basename(image_path), f, "image/jpeg")},
            timeout=60,
        )
    if r.status_code != 200:
        print(_red(f"  ✗ {r.status_code}: {r.text}"))
        return
    data = r.json()
    print(_green(f"  ✓ emotion={data['emotion']}  confidence={data['confidence']:.3f}"))
    for label, score in sorted(data["all_scores"].items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 20)
        print(f"     {label:<15} {bar:<20} {score:.3f}")


def test_speech_emotion(audio_path: str):
    print(_bold(f"\n── /speech-emotion  ({audio_path}) ──────────"))
    with open(audio_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/speech-emotion",
            files={"audio": (os.path.basename(audio_path), f, "audio/wav")},
            timeout=60,
        )
    if r.status_code != 200:
        print(_red(f"  ✗ {r.status_code}: {r.text}"))
        return
    data = r.json()
    print(_green(f"  ✓ emotion={data['emotion']}  confidence={data['confidence']:.3f}"))
    for label, score in sorted(data["all_scores"].items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 20)
        print(f"     {label:<15} {bar:<20} {score:.3f}")


def test_transcript(audio_path: str):
    print(_bold(f"\n── /transcript  ({audio_path}) ──────────────"))
    with open(audio_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/transcript",
            files={"audio": (os.path.basename(audio_path), f, "audio/wav")},
            timeout=120,
        )
    if r.status_code != 200:
        print(_red(f"  ✗ {r.status_code}: {r.text}"))
        return
    data = r.json()
    print(_green(f"  ✓ language={data['language']}"))
    print(f"     transcript: \"{data['transcript']}\"")
    for seg in data["segments"]:
        print(f"     [{seg['start']:.1f}s → {seg['end']:.1f}s] {seg['text']}")


if __name__ == "__main__":
    # Accept optional paths as CLI args: python test_phase1.py face.jpg audio.wav
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    audio_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not image_path or not os.path.exists(image_path):
        print("Usage: python test_phase1.py <path/to/face.jpg> <path/to/speech.wav>")
        print("  → Place a face image and a WAV audio file in the backend/ folder,")
        print("    then run: python test_phase1.py face.jpg speech.wav")
        print("\nRunning health check only …")
        test_health()
        sys.exit(0)

    test_health()
    test_face_emotion(image_path)

    if audio_path and os.path.exists(audio_path):
        test_speech_emotion(audio_path)
        test_transcript(audio_path)
    else:
        print("\nNo audio path provided — skipping speech & transcript tests.")
        print("Usage: python test_phase1.py face.jpg speech.wav")
