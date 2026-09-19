import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'

let landmarkerInstance = null
let initPromise = null

export async function getFaceLandmarker() {
  if (landmarkerInstance) return landmarkerInstance
  if (initPromise) return initPromise

  initPromise = (async () => {
    try {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
      )
      try {
        landmarkerInstance = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
            delegate: 'GPU',
          },
          outputFaceBlendshapes: true,
          runningMode: 'VIDEO',
          numFaces: 1,
        })
      } catch (gpuErr) {
        console.warn('GPU delegate failed, falling back to CPU:', gpuErr)
        landmarkerInstance = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
            delegate: 'CPU',
          },
          outputFaceBlendshapes: true,
          runningMode: 'VIDEO',
          numFaces: 1,
        })
      }
      return landmarkerInstance
    } catch (err) {
      console.warn('Client-side MediaPipe init failed, fallback to server:', err)
      return null
    }
  })()

  return initPromise
}

let lastVideoTimestamp = 0
let lastValidReading = null
let consecutiveMisses = 0

// Resting baseline learner: adapts to each user's unique resting face
let baseline = {
  browDown: 0.14,
  smile: 0.06,
  frown: 0.05,
  eyeWide: 0.05,
  samples: 0,
}

export function resetFaceTracker() {
  lastVideoTimestamp = 0
  lastValidReading = null
  consecutiveMisses = 0
  baseline = {
    browDown: 0.14,
    smile: 0.06,
    frown: 0.05,
    eyeWide: 0.05,
    samples: 0,
  }
}

export function detectFaceEmotionClient(video, landmarker, timestampMs) {
  if (!landmarker || !video || video.readyState < 2) return null
  try {
    const ts = Math.max(Number(timestampMs) || 0, lastVideoTimestamp + 1)
    lastVideoTimestamp = ts

    const results = landmarker.detectForVideo(video, ts)
    if (!results || !results.faceBlendshapes || results.faceBlendshapes.length === 0) {
      consecutiveMisses++
      if (lastValidReading && consecutiveMisses < 12) {
        return lastValidReading
      }
      return null
    }

    consecutiveMisses = 0
    const categories = results.faceBlendshapes[0].categories
    const map = {}
    for (let i = 0; i < categories.length; i++) {
      map[categories[i].categoryName] = categories[i].score
    }

    // ── 1. Action Units extraction ──
    const smile = Math.max(map.mouthSmileLeft || 0, map.mouthSmileRight || 0)
    const cheekSquint = ((map.cheekSquintLeft || 0) + (map.cheekSquintRight || 0)) / 2
    const frown = Math.max(map.mouthFrownLeft || 0, map.mouthFrownRight || 0)
    const browLower = Math.max(map.browDownLeft || 0, map.browDownRight || 0)
    const browInnerUp = map.browInnerUp || 0
    const browOuterUp = ((map.browOuterUpLeft || 0) + (map.browOuterUpRight || 0)) / 2
    const eyeWide = Math.max(map.eyeWideLeft || 0, map.eyeWideRight || 0)
    const eyeSquint = ((map.eyeSquintLeft || 0) + (map.eyeSquintRight || 0)) / 2
    const mouthPress = ((map.mouthPressLeft || 0) + (map.mouthPressRight || 0)) / 2
    const mouthStretch = Math.max(map.mouthStretchLeft || 0, map.mouthStretchRight || 0)
    const sneer = Math.max(map.noseSneerLeft || 0, map.noseSneerRight || 0)
    const lipRaise = Math.max(map.mouthUpperUpLeft || 0, map.mouthUpperUpRight || 0)
    const jawOpen = map.jawOpen || 0

    // ── 2. Automatic Resting Face Calibration (First 25 frames) ──
    if (baseline.samples < 25) {
      baseline.browDown = Math.max(baseline.browDown, Math.min(0.20, browLower))
      baseline.smile = Math.max(baseline.smile, Math.min(0.12, smile))
      baseline.frown = Math.max(baseline.frown, Math.min(0.10, frown))
      baseline.eyeWide = Math.max(baseline.eyeWide, Math.min(0.08, eyeWide))
      baseline.samples++
    }

    // ── 3. Multi-Cue Ekman FACS Gating (prevents false positives) ──

    // ANGRY: Must exceed user's resting brow AND have supporting cues (squint, pressed lips, or sneer)
    const browDownThreshold = Math.max(0.16, baseline.browDown + 0.05)
    const browDownDelta = Math.max(0, browLower - browDownThreshold)
    const angrySupport = (eyeSquint > 0.12 ? 0.3 : 0) + (mouthPress > 0.12 ? 0.35 : 0) + (sneer > 0.10 ? 0.35 : 0)
    let angry = 0
    if (browDownDelta > 0.02 && angrySupport > 0.25) {
      angry = Math.min(browDownDelta * 4.5 + angrySupport, 1.0)
    }

    // FEAR: Wide staring eyes + raised inner brow + mouth stretch (must have at least 2 cues)
    const eyeWideThreshold = Math.max(0.08, baseline.eyeWide + 0.04)
    const eyeWideDelta = Math.max(0, eyeWide - eyeWideThreshold)
    const fearCues = (eyeWideDelta > 0.02 ? 1 : 0) + (browInnerUp > 0.20 ? 1 : 0) + (mouthStretch > 0.12 ? 1 : 0)
    let fear = 0
    if (fearCues >= 2 && jawOpen < 0.35) {
      fear = Math.min(eyeWideDelta * 4.0 + (browInnerUp - 0.15) * 2.5 + mouthStretch * 3.0, 1.0)
    }

    // SURPRISE: Dropped open jaw + raised arched brows
    let surprise = 0
    if (jawOpen > 0.18 && Math.max(browInnerUp, browOuterUp) > 0.16) {
      surprise = Math.min(jawOpen * 2.0 + Math.max(browInnerUp, browOuterUp) * 1.5, 1.0)
    }

    // HAPPY: Genuine smile above resting baseline
    const smileThreshold = Math.max(0.10, baseline.smile + 0.06)
    const smileDelta = Math.max(0, smile - smileThreshold)
    let happy = 0
    if (smileDelta > 0.02) {
      happy = Math.min(smileDelta * 4.5 + cheekSquint * 1.2, 1.0)
    }

    // SAD: Noticeable frown + inner brow contraction
    const frownThreshold = Math.max(0.08, baseline.frown + 0.04)
    const frownDelta = Math.max(0, frown - frownThreshold)
    let sad = 0
    if (frownDelta > 0.02 && browInnerUp > 0.14) {
      sad = Math.min(frownDelta * 3.8 + (browInnerUp - 0.10) * 1.5, 1.0)
    }

    // DISGUST: Clear nose sneer
    let disgust = 0
    if (sneer > 0.14) {
      disgust = Math.min(sneer * 4.2 + lipRaise * 1.8, 1.0)
    }

    // ── 4. Natural Resting Neutral ──
    // If no active expression is being made, the face is reliably NEUTRAL.
    const maxActive = Math.max(happy, surprise, angry, sad, fear, disgust)
    let neutral = Math.min(1.0, Math.max(0.05, Math.exp(-3.5 * maxActive)))

    const raw = {
      happy: happy * 1.0,
      neutral: neutral * (maxActive < 0.15 ? 1.4 : 0.8),
      surprise: surprise * 1.0,
      sad: sad * 1.0,
      fear: fear * 1.0,
      angry: angry * 1.0,
      disgust: disgust * 1.0,
    }

    const total = Object.values(raw).reduce((a, b) => a + b, 0) || 1
    const all_scores = {}
    for (const [k, v] of Object.entries(raw)) {
      all_scores[k] = Math.round((v / total) * 100) / 100
    }

    // EMA smoothing (40/60) for steady, non-jumping readings
    if (lastValidReading && lastValidReading.all_scores) {
      const alpha = 0.4
      for (const k of Object.keys(all_scores)) {
        all_scores[k] = Math.round((alpha * all_scores[k] + (1 - alpha) * (lastValidReading.all_scores[k] || 0)) * 100) / 100
      }
    }

    const sorted = Object.entries(all_scores).sort((a, b) => b[1] - a[1])
    const reading = {
      emotion: sorted[0][0],
      confidence: sorted[0][1],
      all_scores,
      source: 'client-gpu-instant',
    }
    lastValidReading = reading
    return reading
  } catch (err) {
    console.warn('Face tracker tick error:', err)
    return lastValidReading
  }
}
