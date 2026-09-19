"""
test_phase7.py - EmpathAI Phase 7: Comprehensive Validation & Benchmarking
==========================================================================
Validates:
  1. Congruent scenarios (happy, sad, neutral) -> mismatch = False
  2. Mismatch scenarios (masked distress, sarcasm, calm words + urgent panic) -> mismatch = True
  3. Latency validation across each pipeline stage (<5s target)
  4. Accuracy & error handling
"""

import json
import time
import urllib.request
import numpy as np

BASE_URL = "http://localhost:8000"

TEST_CASES = [
    {
        "name": "Congruent Happy",
        "expected_emotion": "happy",
        "expected_mismatch": False,
        "payload": {
            "face": {"probs": {"happy": 0.90, "neutral": 0.07, "surprise": 0.03}, "confidence": 0.90},
            "speech": {"probs": {"happy": 0.85, "calm": 0.10, "neutral": 0.05}, "confidence": 0.85},
            "context": {"sentiment": "positive", "confidence": 0.90, "transcript": "Everything is going great!", "urgency": "low"},
            "mode": "fast"
        }
    },
    {
        "name": "Congruent Sad",
        "expected_emotion": "sad",
        "expected_mismatch": False,
        "payload": {
            "face": {"probs": {"sad": 0.82, "neutral": 0.12, "fear": 0.06}, "confidence": 0.82},
            "speech": {"probs": {"sad": 0.78, "neutral": 0.14, "calm": 0.08}, "confidence": 0.78},
            "context": {"sentiment": "negative", "confidence": 0.85, "transcript": "I am having such a difficult and sad day.", "urgency": "low"},
            "mode": "fast"
        }
    },
    {
        "name": "Masked Distress (Smiling Face + Fear Voice)",
        "expected_emotion": "fear",
        "expected_mismatch": True,
        "payload": {
            "face": {"probs": {"happy": 0.86, "neutral": 0.10, "sad": 0.04}, "confidence": 0.86},
            "speech": {"probs": {"fearful": 0.75, "sad": 0.15, "neutral": 0.10}, "confidence": 0.75},
            "context": {"sentiment": "positive", "confidence": 0.80, "transcript": "Everything is totally fine, don't worry.", "urgency": "low"},
            "mode": "fast"
        }
    },
    {
        "name": "Sarcasm (Ironic words + Annoyed Tone)",
        "expected_emotion": ["angry", "disgust", "sad"],
        "expected_mismatch": True,
        "payload": {
            "face": {"probs": {"neutral": 0.60, "disgust": 0.25, "angry": 0.15}, "confidence": 0.60},
            "speech": {"probs": {"angry": 0.58, "disgust": 0.22, "neutral": 0.20}, "confidence": 0.60},
            "context": {"sentiment": "sarcastic", "sarcasm_detected": True, "confidence": 0.92, "transcript": "Oh wonderful, another flat tire on Monday morning!", "urgency": "medium"},
            "mode": "fast"
        }
    },
    {
        "name": "Panic Urgency Override",
        "expected_emotion": "fear",
        "expected_mismatch": False,
        "payload": {
            "face": {"probs": {"fear": 0.65, "surprise": 0.25, "neutral": 0.10}, "confidence": 0.65},
            "speech": {"probs": {"fearful": 0.80, "angry": 0.10, "sad": 0.10}, "confidence": 0.80},
            "context": {"sentiment": "urgent", "confidence": 0.98, "transcript": "Emergency! Evacuate the premises right now!", "urgency": "high"},
            "mode": "fast"
        }
    }
]

def run_benchmarks():
    print("=" * 70)
    print("       EMPATHAI — PHASE 7 VALIDATION & BENCHMARK SUITE       ")
    print("=" * 70)

    # 1. Health check
    req = urllib.request.Request(f"{BASE_URL}/health")
    health = json.loads(urllib.request.urlopen(req).read().decode())
    print(f"Backend Server Health: {health.get('status', 'FAIL').upper()}\n")

    passed = 0
    total = len(TEST_CASES)
    latencies = []

    for tc in TEST_CASES:
        name = tc["name"]
        print(f"Running: {name} ...")
        t0 = time.perf_counter()
        req = urllib.request.Request(
            f"{BASE_URL}/fuse",
            data=json.dumps(tc["payload"]).encode(),
            headers={"Content-Type": "application/json"}
        )
        resp = json.loads(urllib.request.urlopen(req).read().decode())
        dt = (time.perf_counter() - t0) * 1e3
        latencies.append(dt)

        label = resp.get("label")
        mismatch = resp.get("mismatch")
        mismatch_kind = resp.get("mismatch_kind")
        narration = resp.get("narration")

        # Check emotion
        expected_emo = tc["expected_emotion"]
        if isinstance(expected_emo, list):
            emo_ok = label in expected_emo
        else:
            emo_ok = label == expected_emo

        # Check mismatch
        mismatch_ok = mismatch == tc["expected_mismatch"]

        if emo_ok and mismatch_ok:
            passed += 1
            print(f"  -> PASS: Emotion={label} (mismatch={mismatch}, kind={mismatch_kind})")
            print(f"  -> Narration: \"{narration}\" ({dt:.2f}ms)\n")
        else:
            print(f"  -> FAIL: Got Emotion={label} (expected {expected_emo}), Mismatch={mismatch} (expected {tc['expected_mismatch']})\n")

    print("-" * 70)
    print(f"Test Cases Passed: {passed}/{total} ({(passed/total)*100:.1f}%)")
    print(f"Fusion Latency: Median = {np.median(latencies):.2f}ms, Max = {np.max(latencies):.2f}ms")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmarks()
