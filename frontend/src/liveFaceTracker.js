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

    const smile = ((map.mouthSmileLeft || 0) + (map.mouthSmileRight || 0)) / 2
    const frown = ((map.mouthFrownLeft || 0) + (map.mouthFrownRight || 0)) / 2
    const browDown = ((map.browDownLeft || 0) + (map.browDownRight || 0)) / 2
    const browUp = map.browInnerUp || 0
    const eyeWide = ((map.eyeWideLeft || 0) + (map.eyeWideRight || 0)) / 2
    const eyeSquint = ((map.eyeSquintLeft || 0) + (map.eyeSquintRight || 0)) / 2
    const sneer = ((map.noseSneerLeft || 0) + (map.noseSneerRight || 0)) / 2
    const jawOpen = map.jawOpen || 0

    let happy = Math.min(smile * 2.4, 1.0)
    let surprise = Math.min((jawOpen * 0.8 + eyeWide * 0.8 + browUp * 0.6) / 1.6, 1.0)
    let angry = Math.min((browDown * 2.0 + eyeSquint * 0.7) / 2.0, 1.0)
    let sad = Math.min((frown * 2.2 + browUp * 0.8) / 2.2, 1.0)
    let fear = Math.min((eyeWide * 1.3 + browUp * 0.9 + jawOpen * 0.4) / 2.0, 1.0)
    let disgust = Math.min(sneer * 2.8 + browDown * 0.6, 1.0)

    const activeSum = happy + surprise + angry + sad + fear + disgust
    let neutral = Math.max(0.04, 1.0 - activeSum * 0.9)

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
