"""
context.py
==========
Context Reasoning Layer for EmpathAI.
Uses Gemini Flash to analyze conversational transcripts for emotional tone,
urgency, sarcasm, and latent semantic sentiment.

Supports structured JSON output and graceful fallback on API/network disruption.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import dotenv
from pydantic import BaseModel, Field

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    logger.warning("google.generativeai not installed. Context analysis will run in heuristic mode.")


class ContextAnalysis(BaseModel):
    sentiment: str = Field(
        ...,
        description="Primary emotional sentiment: positive, negative, neutral, urgent, or sarcastic",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence score between 0.0 and 1.0",
    )


class ContextAnalyzer:
    """
    Analyzes spoken dialogue transcripts using Gemini Flash with structured output.
    """

    def __init__(self, model_names: Optional[List[str]] = None):
        self.model_names = model_names or [
            "gemini-3.5-flash-lite",
            "gemini-flash-latest",
            "gemini-2.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-pro-latest"
        ]

        self.client_ready = bool(_GENAI_AVAILABLE and GEMINI_API_KEY)


    def _call_gemini_structured(self, prompt: str) -> Optional[dict]:
        if not self.client_ready:
            return None

        system_instruction = (
            "You are the context reasoning module of EmpathAI, an assistive multimodal system for visually impaired people. "
            "Analyze the user's spoken transcript for emotional tone, latent sentiment, and subtle conversational nuances "
            "(such as sarcasm, irony, distress, or high urgency). "
            "You must return ONLY a JSON object conforming to the requested schema."
        )

        for model_name in self.model_names:
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_instruction,
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.2,
                    },
                )
                response = model.generate_content(prompt)
                if response and response.text:
                    raw_text = response.text.strip()
                    # Clean markdown codeblocks if present
                    if raw_text.startswith("```"):
                        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                        raw_text = re.sub(r"\s*```$", "", raw_text)
                    return json.loads(raw_text)
            except Exception as e:
                logger.warning(f"Gemini call to {model_name} failed: {e}")
                continue

        return None

    def _heuristic_fallback(self, text: str) -> ContextAnalysis:
        """
        Rule-based sentiment fallback when offline or API limit reached.
        """
        lower = text.lower().strip()
        urgent_keywords = ["hurry", "help", "emergency", "fast", "urgent", "quick", "asap", "danger", "now"]
        negative_keywords = ["terrible", "awful", "hate", "bad", "angry", "sad", "upset", "broken", "worst", "fail", "crying"]
        sarcastic_cues = ["oh great", "sure thing", "yeah right", "wonderful, just wonderful", "just perfect", "thanks a lot"]
        positive_keywords = ["great", "love", "wonderful", "happy", "awesome", "excellent", "glad", "delighted", "fantastic"]

        is_urgent = any(kw in lower for kw in urgent_keywords) or ("!" in text and len(text) < 30)
        is_sarcastic = any(cue in lower for cue in sarcastic_cues)
        is_negative = any(kw in lower for kw in negative_keywords)
        is_positive = any(kw in lower for kw in positive_keywords)

        if is_sarcastic:
            sentiment = "sarcastic"
            confidence = 0.85
        elif is_urgent:
            sentiment = "urgent"
            confidence = 0.90
        elif is_negative:
            sentiment = "negative"
            confidence = 0.80
        elif is_positive:
            sentiment = "positive"
            confidence = 0.85
        else:
            sentiment = "neutral"
            confidence = 0.75

        return ContextAnalysis(
            sentiment=sentiment,
            confidence=confidence,
        )

    def analyze(self, text: str) -> ContextAnalysis:
        if not text or not text.strip():
            return ContextAnalysis(
                sentiment="neutral",
                confidence=0.5,
            )

        prompt = f"""
Analyze the following spoken transcript:
\"\"\"{text}\"\"\"

Return ONLY a JSON object strictly matching this schema:
{{
  "sentiment": "<one of 'positive', 'neutral', 'negative', 'urgent', 'sarcastic'>",
  "confidence": <float between 0.0 and 1.0>
}}
"""
        raw_json = self._call_gemini_structured(prompt)
        if raw_json:
            try:
                return ContextAnalysis.model_validate(raw_json)
            except Exception as e:
                logger.warning(f"Failed to validate Gemini JSON with Pydantic: {e}. Raw: {raw_json}")

        return self._heuristic_fallback(text)


# ── Global singleton & drop-in entrypoint ─────────────────────────────────────

_GLOBAL_CONTEXT_ANALYZER: Optional[ContextAnalyzer] = None

def analyze_context(text: str) -> dict:
    """Drop-in functional entrypoint for FastAPI and pipeline orchestration."""
    global _GLOBAL_CONTEXT_ANALYZER
    if _GLOBAL_CONTEXT_ANALYZER is None:
        _GLOBAL_CONTEXT_ANALYZER = ContextAnalyzer()
    res = _GLOBAL_CONTEXT_ANALYZER.analyze(text)
    return res.model_dump()


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        "Everything is running smoothly and I'm feeling really happy about our progress today.",
        "Quick! We need to stop the server immediately before data gets corrupted!",
        "Oh wonderful, another flat tire on Monday morning. Just what I needed!",
        "The current temperature outside is twenty-two degrees Celsius.",
    ]

    analyzer = ContextAnalyzer()
    print("Testing ContextAnalyzer with sample inputs:")
    for text in test_cases:
        result = analyzer.analyze(text)
        print(f"\nTranscript: \"{text}\"")
        print(f"  -> Sentiment: {result.sentiment} (confidence: {result.confidence:.2f})")
        print(f"  -> Urgency: {result.urgency} | Sarcasm: {result.sarcasm_detected}")
        print(f"  -> Explanation: {result.explanation}")

    print("\nAll context self-tests completed.")

