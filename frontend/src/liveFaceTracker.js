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

export function resetFaceTracker() {
  lastVideoTimestamp = 0
  lastValidReading = null
  consecutiveMisses = 0
}

export function detectFaceEmotionClient(video, landmarker, timestampMs) {
  if (!landmarker || !video || video.readyState < 2) return null
  try {
    // MediaPipe strictly requires timestamps to be strictly increasing: ts > lastVideoTimestamp
    const ts = Math.max(Number(timestampMs) || 0, lastVideoTimestamp + 1)
    lastVideoTimestamp = ts

    const results = landmarker.detectForVideo(video, ts)
    if (!results || !results.faceBlendshapes || results.faceBlendshapes.length === 0) {
      consecutiveMisses++
      // If face is momentarily lost (blink, head tilt), retain previous valid reading
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

    // ── 1. Action Units extraction from MediaPipe Blendshapes ──
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

    // ── 2. Scientific Ekman FACS Heuristics with Resting Floor Subtraction ──
    // ANGRY: AU4 (corrugator brow lowerer) + AU7 (lid tightener) + AU24 (lip press) + AU9 (nose sneer)
    const browDownActive = Math.max(0, browLower - 0.02) * 5.2
    const angryRaw = browDownActive * 0.90 + eyeSquint * 1.6 + mouthPress * 2.2 + sneer * 1.5
    let angry = Math.min(Math.max(0, angryRaw), 1.0)

    // FEAR: AU5 (wide staring eyes) + AU1 (inner brow raiser) + AU20 (mouth horizontal stretch)
    // Differentiated from Surprise: Fear has horizontal mouth stretch / tense eyes, surprise has open dropped jaw
    const eyeWideActive = Math.max(0, eyeWide - 0.015) * 5.0
    const innerBrowActive = Math.max(0, browInnerUp - 0.02) * 3.6
    const fearRaw = eyeWideActive * 0.75 + innerBrowActive * 0.45 + mouthStretch * 3.5 - Math.max(0, jawOpen - 0.20) * 1.2
    let fear = Math.min(Math.max(0, fearRaw), 1.0)

    // SURPRISE: AU26 (dropped open jaw) + AU1+2 (high arched brows) + AU5 (wide eyes)
    let surprise = Math.min(jawOpen * 1.8 + Math.max(browInnerUp, browOuterUp) * 1.6 + eyeWideActive * 0.6, 1.0)

    // HAPPY: AU12 (zygomaticus major smile) + AU6 (orbicularis oculi cheek raise - Duchenne marker)
    let happy = Math.min(smile * 4.4 + cheekSquint * 1.2, 1.0)

    // SAD: AU15 (depressor anguli oris frown) + AU1 (inner brow raise)
    let sad = Math.min(Math.max(0, frown - 0.02) * 4.2 + innerBrowActive * 0.6, 1.0)

    // DISGUST: AU9 (levator labii superioris nose sneer) + AU10 (upper lip raiser)
    let disgust = Math.min(sneer * 4.6 + lipRaise * 2.2, 1.0)

    // ── 3. Exponential Neutral Decay ──
    // In human psychology, Neutral is the absence of active cues.
    // When ANY emotion fires, Neutral decays exponentially: exp(-4.2 * max)
    const maxActive = Math.max(happy, surprise, angry, sad, fear, disgust)
    let neutral = Math.min(1.0, Math.max(0.02, Math.exp(-4.2 * maxActive)))

    const raw = { happy, neutral, surprise, sad, fear, angry, disgust }
    const total = Object.values(raw).reduce((a, b) => a + b, 0) || 1
    const all_scores = {}
    for (const [k, v] of Object.entries(raw)) {
      all_scores[k] = Math.round((v / total) * 100) / 100
    }

    // Smooth with previous frame (EMA 50/50) for buttery-smooth non-flickering bars
    if (lastValidReading && lastValidReading.all_scores) {
      const alpha = 0.5
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
