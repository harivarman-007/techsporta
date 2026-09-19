# Phase 3 — Fusion Layer

- [x] Design fusion input format: concatenated [face_label_vector, speech_label_vector, context_sentiment_vector] (38-dim feature vector with presence flags, confidence scores, and pairwise valence conflict features)
- [x] Author a small synthetic labeled dataset of modality-label combinations → expected fused label + mismatch flag (document this as a known limitation — no large real fusion dataset exists; generated 40k train + 8k shifted validation across 14 scenarios)
- [x] Build and train a small PyTorch MLP (3 layers) on the synthetic set → outputs final emotion label + binary mismatch flag (val acc 83%, mismatch precision 94%, recall 92%, <1ms CPU latency)
- [x] Implement a parallel Gemini-based fusion path: send all three modality outputs to Gemini Flash with structured output, ask it to reconcile into a final label + mismatch flag
- [x] Run both fusion paths against the same test combinations, log results side by side for comparison


## Deliverable
Two working fusion approaches (trained MLP + LLM reasoning), both tested against the same sample modality combinations, results logged for comparison.

## Deviations
- Python runtime: Running on Python 3.13.1 on host system.
- Synthetic Training Set: MLP trained on 40k synthetic vectors across 14 scenarios due to absence of real paired 3-channel dataset (accurately documented as synthetic validation only).
- Comparison Log: Side-by-side comparison between `MLPFusionHead` and Gemini Flash reconciler logged to `backend/fusion_comparison.log`.
