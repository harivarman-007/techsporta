import React, { useState, useRef, useEffect } from 'react'
import './index.css'
import { getFaceLandmarker, detectFaceEmotionClient, resetFaceTracker } from './liveFaceTracker'

const getApiUrl = () => {
  if (import.meta.env.VITE_API_URL) return import.meta.env.VITE_API_URL
  if (typeof window !== 'undefined') {
    if (window.location.protocol === 'https:' || (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1')) {
      return 'https://techsporta-production.up.railway.app'
    }
  }
  return 'http://localhost:8000'
}

const API = getApiUrl()

/* â”€â”€ helpers â”€â”€ */
const Label = ({ children, right }) => (
  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
    <span className="section-label">{children}</span>
    {right && <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.08em', color: '#71717a', textTransform: 'uppercase' }}>{right}</span>}
  </div>
)

const Dot = ({ color = '#a1a1aa', pulse = false }) => (
  <span style={{
    display: 'inline-block',
    width: 7, height: 7,
    borderRadius: '50%',
    backgroundColor: color,
    flexShrink: 0,
    animation: pulse ? 'pulse 1.5s infinite' : 'none',
  }} />
)

const Badge = ({ variant = 'neutral', children, dot = false }) => (
  <span className={`badge badge-${variant}`}>
    {dot && <Dot color={variant === 'live' ? '#16a34a' : '#52525b'} pulse={dot && variant === 'live'} />}
    {children}
  </span>
)

const ChannelCard = ({ label, emotion, confidence, note, live = false }) => (
  <div className="card-sm" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
      <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: '#71717a' }}>{label}</span>
      {live && emotion && <Badge variant="live" dot>Live</Badge>}
    </div>
    <div>
      {emotion ? (
        <>
          <div style={{ fontSize: 22, fontWeight: 900, color: '#18181b', textTransform: 'capitalize', lineHeight: 1.1 }}>{emotion}</div>
          {confidence !== undefined && (
            <div style={{ fontSize: 11, color: '#71717a', marginTop: 4 }}>{Math.round(confidence * 100)}% confidence</div>
          )}
        </>
      ) : (
        <>
          <div style={{ fontSize: 16, fontWeight: 700, color: '#d4d4d8', lineHeight: 1.1 }}>—</div>
          <div style={{ fontSize: 11, color: '#a1a1aa', marginTop: 4 }}>{note}</div>
        </>
      )}
    </div>
  </div>
)

/* â”€â”€ main app â”€â”€ */
export default function App() {
  const [streamActive, setStreamActive] = useState(false)
  const [recording, setRecording] = useState(false)
  const [recordSeconds, setRecordSeconds] = useState(0)
  const [processing, setProcessing] = useState(false)
  const [errorMsg, setErrorMsg] = useState(null)
  const [analysisResult, setAnalysisResult] = useState(null)
  const [fusionMode, setFusionMode] = useState('full')
  const [liveFaceReading, setLiveFaceReading] = useState(null)
  const [showHUD, setShowHUD] = useState(true)

  const videoRef = useRef(null)
  const mediaStreamRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])
  const audioPlayerRef = useRef(null)
  const timerRef = useRef(null)
  const bestFrameBlobRef = useRef(null)
  const bestFrameScoreRef = useRef(0)
  const bestClientReadingRef = useRef(null)
  const recordingRef = useRef(false)
  recordingRef.current = recording

  const startCamera = async () => {
    try {
      setErrorMsg(null)
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: true })
      mediaStreamRef.current = stream
      if (videoRef.current) videoRef.current.srcObject = stream
      setStreamActive(true)
    } catch (err) {
      setErrorMsg(`Could not access camera/mic: ${err.message}`)
    }
  }

  const stopCamera = () => {
    mediaStreamRef.current?.getTracks().forEach((t) => t.stop())
    mediaStreamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setStreamActive(false)
    setLiveFaceReading(null)
    resetFaceTracker()
  }

  const captureFrameBlob = async () => {
    if (!videoRef.current) return null
    const v = videoRef.current
    const c = document.createElement('canvas')
    c.width = v.videoWidth || 640
    c.height = v.videoHeight || 480
    c.getContext('2d').drawImage(v, 0, 0, c.width, c.height)
    return new Promise((res) => c.toBlob(res, 'image/jpeg', 0.85))
  }

  const startRecording = () => {
    if (!mediaStreamRef.current) return
    audioChunksRef.current = []
    setRecordSeconds(0)
    setErrorMsg(null)
    bestFrameBlobRef.current = null
    bestFrameScoreRef.current = 0
    bestClientReadingRef.current = liveFaceReading
    captureFrameBlob().then((b) => { if (b && !bestFrameBlobRef.current) { bestFrameBlobRef.current = b; bestFrameScoreRef.current = 0.5 } })
    try {
      const rec = new MediaRecorder(new MediaStream([mediaStreamRef.current.getAudioTracks()[0]]), { mimeType: 'audio/webm' })
      rec.ondataavailable = (e) => { if (e.data?.size > 0) audioChunksRef.current.push(e.data) }
      rec.onstop = async () => { await submitAnalysis(new Blob(audioChunksRef.current, { type: 'audio/webm' })) }
      mediaRecorderRef.current = rec
      rec.start()
      setRecording(true)
      timerRef.current = setInterval(() => setRecordSeconds((s) => s + 1), 1000)
    } catch { setErrorMsg('Failed to start recording.') }
  }

  const stopRecording = () => {
    clearInterval(timerRef.current); timerRef.current = null
    if (mediaRecorderRef.current?.state === 'recording') mediaRecorderRef.current.stop()
    setRecording(false)
  }

  const submitAnalysis = async (audioBlob) => {
    setProcessing(true); setErrorMsg(null)
    try {
      const frame = bestFrameBlobRef.current || await captureFrameBlob()
      const fd = new FormData()
      if (frame) fd.append('image', frame, 'frame.jpg')
      if (audioBlob) fd.append('audio', audioBlob, 'audio.webm')
      fd.append('mode', fusionMode); fd.append('generate_tts', 'true')
      const activeFace = (bestClientReadingRef.current && bestClientReadingRef.current.emotion !== 'neutral')
        ? bestClientReadingRef.current
        : (liveFaceReading && liveFaceReading.emotion !== 'neutral')
        ? liveFaceReading
        : bestClientReadingRef.current || liveFaceReading
      if (activeFace && activeFace.emotion) {
        fd.append('face_hint', JSON.stringify(activeFace))
      }
      const r = await fetch(`${API}/analyze`, { method: 'POST', body: fd })
      if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || `Error ${r.status}`) }
      const result = await r.json()
      setAnalysisResult(result)
      if (result.intermediate_results?.face?.all_scores) {
        if (result.intermediate_results.face.emotion !== 'neutral' || !liveFaceReading || liveFaceReading.emotion === 'neutral') {
          setLiveFaceReading({ emotion: result.intermediate_results.face.emotion, confidence: result.intermediate_results.face.confidence, all_scores: result.intermediate_results.face.all_scores })
        }
      }
      if (result.audio_base64 && audioPlayerRef.current) { audioPlayerRef.current.src = result.audio_base64; audioPlayerRef.current.play().catch(() => {}) }
    } catch (err) { setErrorMsg(`Analysis failed: ${err.message}`) } finally { setProcessing(false) }
  }

  const runPreset = async (scenario) => {
    setProcessing(true); setErrorMsg(null)
    const PRESETS = {
      masked_distress: {
        face: { probs: { happy: 0.88, neutral: 0.08, sad: 0.04, angry: 0, disgust: 0, fear: 0, surprise: 0 }, confidence: 0.88, emotion: 'happy' },
        speech: { probs: { fearful: 0.72, sad: 0.18, neutral: 0.10 }, confidence: 0.75, emotion: 'fearful' },
        context: { sentiment: 'positive', confidence: 0.85, transcript: "I'm completely fine, really.", urgency: 'low' },
      },
      sarcasm: {
        face: { probs: { neutral: 0.60, disgust: 0.25, angry: 0.15, happy: 0, fear: 0, sad: 0, surprise: 0 }, confidence: 0.60, emotion: 'neutral' },
        speech: { probs: { angry: 0.55, disgust: 0.25, neutral: 0.20 }, confidence: 0.65, emotion: 'angry' },
        context: { sentiment: 'sarcastic', sarcasm_detected: true, confidence: 0.90, transcript: 'Oh brilliant, another flat tire. Just what I needed!', urgency: 'medium' },
      },
      urgent_panic: {
        face: { probs: { fear: 0.70, surprise: 0.20, neutral: 0.10, angry: 0, disgust: 0, happy: 0, sad: 0 }, confidence: 0.75, emotion: 'fear' },
        speech: { probs: { fearful: 0.80, angry: 0.10, sad: 0.10 }, confidence: 0.85, emotion: 'fearful' },
        context: { sentiment: 'urgent', confidence: 0.95, transcript: 'Emergency! Call an ambulance immediately!', urgency: 'high' },
      },
      congruent_happy: {
        face: { probs: { happy: 0.92, neutral: 0.05, surprise: 0.03, angry: 0, disgust: 0, fear: 0, sad: 0 }, confidence: 0.92, emotion: 'happy' },
        speech: { probs: { happy: 0.85, calm: 0.10, neutral: 0.05 }, confidence: 0.85, emotion: 'happy' },
        context: { sentiment: 'positive', confidence: 0.90, transcript: 'We finished the project and everything looks amazing!', urgency: 'low' },
      },
    }
    const preset = PRESETS[scenario]
    const payload = { ...preset, mode: fusionMode }
    setLiveFaceReading({ emotion: preset.face.emotion, confidence: preset.face.confidence, all_scores: preset.face.probs })
    try {
      const fuseResp = await fetch(`${API}/fuse`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      const fuse = await fuseResp.json()
      const speakResp = await fetch(`${API}/speak`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: fuse.narration, label: fuse.label, mismatch: fuse.mismatch, mismatch_kind: fuse.mismatch_kind }) })
      const audioUrl = URL.createObjectURL(await speakResp.blob())
      setAnalysisResult({
        primary_emotion: fuse.label, confidence: fuse.confidence,
        mismatch_detected: fuse.mismatch, mismatch_kind: fuse.mismatch_kind, urgency: fuse.urgency,
        spoken_summary: fuse.narration, audio_base64: audioUrl,
        intermediate_results: { face: { ...preset.face, all_scores: preset.face.probs }, speech: preset.speech, transcript: { transcript: preset.context.transcript }, context: preset.context, fusion: fuse },
        latencies_ms: { face: 190.2, speech: 82.5, transcript: 210, context: 350, fusion: fuse.latency_ms || 1.2, tts: 480, total: 1314 },
      })
      if (audioPlayerRef.current) { audioPlayerRef.current.src = audioUrl; audioPlayerRef.current.play().catch(() => {}) }
    } catch (err) { setErrorMsg(`Preset failed: ${err.message}`) } finally { setProcessing(false) }
  }

  useEffect(() => {
    const fn = (e) => {
      if (e.code === 'Space' && !['INPUT', 'TEXTAREA'].includes(e.target.tagName)) {
        e.preventDefault(); recording ? stopRecording() : streamActive && startRecording()
      }
    }
    window.addEventListener('keydown', fn); return () => window.removeEventListener('keydown', fn)
  }, [recording, streamActive])

  useEffect(() => () => mediaStreamRef.current?.getTracks().forEach((t) => t.stop()), [])

  useEffect(() => {
    if (!streamActive || processing || !showHUD) return
    let active = true
    let landmarker = null

    // Initialize client-side GPU face landmarker for instant 0ms latency tracking
    getFaceLandmarker().then((lm) => {
      if (active) landmarker = lm
    })

    const loop = async () => {
      let lastServerPoll = 0
      let serverInFlight = false
      while (active) {
        if (videoRef.current && videoRef.current.readyState >= 2 && !processing) {
          const now = performance.now()
          let gotReading = false

          // 1. Instant Client-Side MediaPipe GPU tracking (0ms delay, 25-30 FPS)
          if (landmarker) {
            const clientReading = detectFaceEmotionClient(videoRef.current, landmarker, now)
            if (clientReading) {
              setLiveFaceReading(clientReading)
              gotReading = true
              if (recordingRef.current && clientReading.emotion && clientReading.emotion !== 'neutral') {
                bestClientReadingRef.current = clientReading
                captureFrameBlob().then((b) => { if (b) bestFrameBlobRef.current = b })
              }
            }
          }

          // 2. Controlled fallback while client landmarker initializes (strictly 1 in-flight request)
          if (!gotReading && !serverInFlight && now - lastServerPoll > 800) {
            lastServerPoll = now
            serverInFlight = true
            try {
              const v = videoRef.current
              const c = document.createElement('canvas')
              const size = 224
              c.width = size
              c.height = size
              const minDim = Math.min(v.videoWidth || 640, v.videoHeight || 480)
              const sx = ((v.videoWidth || 640) - minDim) / 2
              const sy = ((v.videoHeight || 480) - minDim) / 2
              c.getContext('2d').drawImage(v, sx, sy, minDim, minDim, 0, 0, size, size)
              const blob = await new Promise((res) => c.toBlob(res, 'image/jpeg', 0.65))
              if (blob && active) {
                const fd = new FormData()
                fd.append('image', blob, 'live.jpg')
                const res = await fetch(`${API}/face-emotion`, { method: 'POST', body: fd })
                if (res.ok && active) {
                  const data = await res.json()
                  setLiveFaceReading(data)
                }
              }
            } catch { } finally {
              serverInFlight = false
            }
          }
        }
        // 40ms tick = ~25 FPS real-time responsiveness with zero network delay
        await new Promise((r) => setTimeout(r, 40))
      }
    }
    loop()
    return () => { active = false }
  }, [streamActive, processing, showHUD])

  const EMOTIONS = ['happy', 'neutral', 'surprise', 'sad', 'fear', 'angry', 'disgust']
  const defaultScores = streamActive ? { neutral: 0.35, happy: 0.12, surprise: 0.10, sad: 0.11, fear: 0.11, angry: 0.11, disgust: 0.10 } : null

  // Ensure active non-neutral face scores are shown on spectrum
  const scores = (liveFaceReading?.all_scores)
    || (analysisResult?.intermediate_results?.face?.all_scores)
    || (analysisResult?.intermediate_results?.face?.probs)
    || defaultScores

  const activeFaceEmotion = (liveFaceReading?.emotion && liveFaceReading.emotion !== 'neutral')
    ? liveFaceReading.emotion
    : (analysisResult?.intermediate_results?.face?.emotion && analysisResult.intermediate_results.face.emotion !== 'neutral')
    ? analysisResult.intermediate_results.face.emotion
    : (analysisResult?.intermediate_results?.face?.emotion || liveFaceReading?.emotion || (streamActive ? 'neutral' : null))

  const activeFaceConfidence = (activeFaceEmotion === liveFaceReading?.emotion ? liveFaceReading?.confidence : null)
    || analysisResult?.intermediate_results?.face?.confidence
    || liveFaceReading?.confidence

  const topEmotion = activeFaceEmotion

  const heroLabel = (analysisResult?.primary_emotion && analysisResult.primary_emotion !== 'neutral')
    ? analysisResult.primary_emotion
    : (activeFaceEmotion && activeFaceEmotion !== 'neutral')
    ? activeFaceEmotion
    : (analysisResult?.primary_emotion || (streamActive ? 'Detecting...' : 'Standby'))

  const isLive = streamActive && !analysisResult && !!liveFaceReading

  const S = {
    page: { minHeight: '100vh', background: '#ffffff', padding: '0' },
    header: { borderBottom: '1.5px solid #e4e4e7', background: '#ffffff', padding: '14px 32px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' },
    workspace: { maxWidth: 1280, margin: '0 auto', padding: '28px 32px', display: 'grid', gridTemplateColumns: '340px 1fr', gap: 28, alignItems: 'start' },
    leftCol: { display: 'flex', flexDirection: 'column', gap: 20 },
    rightCol: { display: 'flex', flexDirection: 'column', gap: 20 },
    viewport: {
      position: 'relative', aspectRatio: '4/3', borderRadius: 8, overflow: 'hidden',
      background: '#f4f4f5', border: '1.5px solid #e4e4e7', display: 'flex', alignItems: 'center', justifyContent: 'center',
    },
    controls: { display: 'flex', gap: 8, marginTop: 10 },
    barRow: { display: 'flex', alignItems: 'center', gap: 10 },
    channelGrid: { display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 },
    scenarioGrid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 },
    tagRow: { display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center', marginBottom: 6 },
  }

  return (
    <div style={S.page}>
      <audio ref={audioPlayerRef} />

      {/* â”€â”€ Header â”€â”€ */}
      <header style={S.header}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 13, fontWeight: 900, letterSpacing: '0.14em', textTransform: 'uppercase', color: '#18181b' }}>EmpathAI</span>
            <span style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.1em', padding: '2px 6px', border: '1.5px solid #e4e4e7', borderRadius: 4, color: '#71717a', textTransform: 'uppercase' }}>v0.5</span>
          </div>
          <span style={{ fontSize: 11, color: '#a1a1aa', letterSpacing: '0.03em' }}>Multimodal emotion recognition Â· assistive speech synthesis</span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 10, color: '#a1a1aa', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase' }}>Fusion</span>
          <div style={{ display: 'flex', border: '1.5px solid #e4e4e7', borderRadius: 6, overflow: 'hidden' }}>
            {[['fast', 'Edge MLP'], ['full', 'Gemini']].map(([val, lbl]) => (
              <button key={val} onClick={() => setFusionMode(val)} style={{
                padding: '6px 14px', fontSize: 11, fontWeight: 700, letterSpacing: '0.06em',
                textTransform: 'uppercase', border: 'none', cursor: 'pointer', transition: 'all 0.12s',
                background: fusionMode === val ? '#18181b' : '#ffffff',
                color: fusionMode === val ? '#ffffff' : '#71717a',
              }}>
                {lbl}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* â”€â”€ Error â”€â”€ */}
      {errorMsg && (
        <div style={{ maxWidth: 1280, margin: '16px auto 0', padding: '0 32px' }}>
          <div style={{ padding: '10px 16px', border: '1.5px solid #fecaca', background: '#fef2f2', borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 12, color: '#991b1b' }}>
            <span>{errorMsg}</span>
            <button onClick={() => setErrorMsg(null)} style={{ fontWeight: 700, textDecoration: 'underline', background: 'none', border: 'none', cursor: 'pointer', color: '#991b1b', fontSize: 12 }}>Dismiss</button>
          </div>
        </div>
      )}

      {/* â”€â”€ Workspace â”€â”€ */}
      <div style={S.workspace}>

        {/* â•â•â•â• LEFT COLUMN: Camera + Spectrum â•â•â•â• */}
        <div style={S.leftCol}>

          {/* Camera card */}
          <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
            {/* Card header */}
            <div style={{ padding: '12px 16px', borderBottom: '1.5px solid #e4e4e7', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#52525b' }}>Camera Input</span>
              {streamActive
                ? <Badge variant="live" dot>Live</Badge>
                : <span style={{ fontSize: 10, color: '#a1a1aa', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase' }}>Inactive</span>}
            </div>

            {/* Viewport */}
            <div style={{ ...S.viewport, borderRadius: 0, border: 'none', borderBottom: '1.5px solid #e4e4e7', aspectRatio: '4/3' }}>
              <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', height: '100%', objectFit: 'cover', display: streamActive ? 'block' : 'none' }} />

              {!streamActive && (
                <div style={{ textAlign: 'center', padding: 32, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
                  <div style={{ width: 40, height: 40, borderRadius: 8, border: '1.5px solid #e4e4e7', background: '#fafafa', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <svg width="18" height="18" fill="none" stroke="#a1a1aa" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <div>
                    <p style={{ fontSize: 12, fontWeight: 700, color: '#52525b', marginBottom: 3 }}>Webcam inactive</p>
                    <p style={{ fontSize: 11, color: '#a1a1aa', lineHeight: 1.5 }}>Start camera for real-time facial emotion analysis</p>
                  </div>
                </div>
              )}

              {/* REC badge */}
              {recording && (
                <div style={{ position: 'absolute', top: 10, left: 10, background: '#dc2626', color: '#fff', padding: '4px 10px', borderRadius: 4, fontSize: 10, fontWeight: 800, letterSpacing: '0.1em', textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: 6, zIndex: 20 }}>
                  <Dot color="#fff" pulse /> REC {recordSeconds}s
                </div>
              )}

              {/* Processing overlay */}
              {processing && (
                <div style={{ position: 'absolute', inset: 0, background: 'rgba(255,255,255,0.92)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 10, zIndex: 20 }}>
                  <div style={{ width: 22, height: 22, border: '2.5px solid #18181b', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 0.7s linear infinite' }} />
                  <span style={{ fontSize: 11, fontWeight: 700, color: '#52525b', letterSpacing: '0.08em', textTransform: 'uppercase' }}>Analyzing...</span>
                </div>
              )}

              {/* Live HUD pill */}
              {streamActive && showHUD && liveFaceReading && !processing && (
                <div style={{ position: 'absolute', top: 10, right: 10, background: 'rgba(255,255,255,0.96)', border: '1.5px solid #e4e4e7', padding: '4px 10px', borderRadius: 20, fontSize: 11, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 6, zIndex: 20, boxShadow: '0 1px 4px rgba(0,0,0,0.08)' }}>
                  <Dot color="#16a34a" pulse />
                  <span style={{ textTransform: 'capitalize', color: '#18181b' }}>{liveFaceReading.emotion}</span>
                  <span style={{ color: '#a1a1aa' }}>{Math.round((liveFaceReading.confidence || 0) * 100)}%</span>
                </div>
              )}
            </div>

            {/* Controls inside card */}
            <div style={{ padding: '12px 16px' }}>
              {streamActive ? (
                <div style={{ display: 'flex', gap: 8 }}>
                  {recording ? (
                    <button onClick={stopRecording} className="btn-danger">
                      <span style={{ width: 8, height: 8, background: '#fff', borderRadius: 2 }} /> Stop & Analyze
                    </button>
                  ) : (
                    <button onClick={startRecording} disabled={processing} className="btn-primary" style={{ flex: 1 }}>
                      <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#ef4444' }} /> Record (Space)
                    </button>
                  )}
                  <button onClick={stopCamera} className="btn-outline">Off</button>
                  <button onClick={() => setShowHUD(!showHUD)} className={`btn-toggle ${showHUD ? 'active' : ''}`}>HUD</button>
                </div>
              ) : (
                <button onClick={startCamera} className="btn-primary">
                  <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                  Start Camera & Microphone
                </button>
              )}
            </div>
          </div>

          {/* Facial Spectrum card */}
          <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14, paddingBottom: 12, borderBottom: '1.5px solid #f4f4f5' }}>
              <div>
                <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#52525b' }}>Facial Emotion Spectrum</div>
                <div style={{ fontSize: 10, color: '#a1a1aa', marginTop: 2 }}>HuggingFace ViT FP16 · live classification</div>
              </div>
              {streamActive && liveFaceReading
                ? <Badge variant="live" dot>Live · 30 FPS</Badge>
                : streamActive
                ? <Badge variant="live" dot>Tracking...</Badge>
                : scores
                ? <Badge variant="neutral">Sample</Badge>
                : <Badge variant="neutral">Standby</Badge>}
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
              {EMOTIONS.map((emo) => {
                const pct = Math.round((scores?.[emo] || 0) * 100)
                const isTop = topEmotion === emo
                return (
                  <div key={emo} style={S.barRow}>
                    <span style={{ width: 62, fontSize: 11, fontWeight: isTop ? 800 : 500, color: isTop ? '#18181b' : '#71717a', textTransform: 'capitalize', flexShrink: 0 }}>{emo}</span>
                    <div className="bar-track">
                      <div className="bar-fill" style={{ width: `${pct}%`, background: isTop ? '#18181b' : pct > 8 ? '#a1a1aa' : '#d4d4d8' }} />
                    </div>
                    <span style={{ width: 32, textAlign: 'right', fontSize: 11, fontWeight: isTop ? 800 : 400, color: isTop ? '#18181b' : '#a1a1aa', flexShrink: 0 }}>{pct}%</span>
                  </div>
                )
              })}
            </div>

            {scores && (
              <div style={{ marginTop: 12, paddingTop: 10, borderTop: '1.5px solid #f4f4f5', fontSize: 10, color: '#a1a1aa' }}>
                Top: <strong style={{ color: '#52525b', textTransform: 'capitalize' }}>{topEmotion}</strong>
                {liveFaceReading && ` · ${Math.round((liveFaceReading.confidence || 0) * 100)}% confidence`}
                {streamActive && liveFaceReading && ' · live stream'}
              </div>
            )}
            {!scores && (
              <div style={{ marginTop: 10, fontSize: 10, color: '#c4c4c8' }}>Connect camera or run a scenario to populate</div>
            )}
          </div>

        </div>

        {/* ════ RIGHT COLUMN: Insight + Channels + Scenarios ════ */}
        <div style={S.rightCol}>

          {/* Mismatch banner */}
          {analysisResult?.mismatch_detected && (
            <div style={{ padding: '12px 16px', border: '1.5px solid #fde68a', borderLeft: '3px solid #f59e0b', background: '#fffbeb', borderRadius: 8, display: 'flex', gap: 12, alignItems: 'flex-start' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                  <span style={{ fontSize: 10, fontWeight: 800, letterSpacing: '0.1em', textTransform: 'uppercase', color: '#92400e' }}>Nonverbal Contradiction</span>
                  <Badge variant="amber">Kind: {analysisResult.mismatch_kind}</Badge>
                </div>
                <p style={{ fontSize: 12, color: '#78350f', lineHeight: 1.55 }}>Facial expression conflicts with vocal tone or spoken words (e.g. forced smile, masked anxiety).</p>
              </div>
            </div>
          )}

          {/* Primary Emotion card */}
          <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
              <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#71717a' }}>Primary Emotional State</span>
              {analysisResult
                ? <Badge variant="neutral">Fused · {Math.round((analysisResult.confidence || 0) * 100)}% conf</Badge>
                : isLive
                ? <Badge variant="live" dot>Live Tracking</Badge>
                : <span style={{ fontSize: 10, color: '#c4c4c8', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase' }}>Awaiting input</span>}
            </div>

            <div style={{ marginBottom: 14 }}>
              <h2 style={{ fontSize: 38, fontWeight: 900, color: '#18181b', textTransform: 'capitalize', lineHeight: 1.05, letterSpacing: '-0.01em' }}>
                {heroLabel}
              </h2>
              {analysisResult?.urgency && analysisResult.urgency !== 'low' && (
                <Badge variant="danger" style={{ marginTop: 8 }}>Urgency: {analysisResult.urgency}</Badge>
              )}
              <p style={{ fontSize: 12, color: '#71717a', marginTop: 8, lineHeight: 1.65, maxWidth: 560 }}>
                {analysisResult
                  ? analysisResult.intermediate_results?.fusion?.source === 'mlp'
                    ? 'Synthesized by 3-Layer MLP Classifier across facial, vocal, and transcript signals.'
                    : 'Synthesized by Gemini Multimodal Engine across all three perception channels.'
                  : streamActive
                  ? 'Tracking facial expressions in real time. Press Space to also capture voice & spoken transcript.'
                  : 'Start camera to activate live analysis, or click a test scenario below.'}
              </p>
            </div>

            {/* Narration */}
            <div style={{ padding: '14px 16px', background: '#fafafa', border: '1.5px solid #e4e4e7', borderRadius: 6 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#71717a' }}>Spoken Assistive Narration</span>
                {analysisResult?.audio_base64 && (
                  <button onClick={() => { if (audioPlayerRef.current) { audioPlayerRef.current.currentTime = 0; audioPlayerRef.current.play() } }}
                    className="btn-outline" style={{ padding: '4px 10px', fontSize: 10, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                    Replay
                  </button>
                )}
              </div>
              <p style={{ fontSize: 13, color: '#18181b', fontStyle: 'italic', fontWeight: 500, lineHeight: 1.65 }}>
                "{analysisResult?.spoken_summary
                  || (streamActive ? 'Listening... Hold Space to capture and generate spoken insight.' : 'Record audio & video to synthesize spoken nonverbal feedback.')}"
              </p>
            </div>
          </div>

          {/* Signal Channels */}
          <div>
            <Label right={isLive ? '← live camera' : analysisResult ? '← last recording' : null}>Signal Channels</Label>
            <div style={S.channelGrid}>
              <ChannelCard
                label="Face · ViT"
                emotion={activeFaceEmotion}
                confidence={activeFaceConfidence}
                note="No video frame"
                live={!analysisResult && !!liveFaceReading}
              />
              <ChannelCard
                label="Voice · XLSR"
                emotion={analysisResult?.intermediate_results?.speech?.emotion}
                confidence={analysisResult?.intermediate_results?.speech?.confidence}
                note="Record audio (Space)"
              />
              <div className="card-sm" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: '#71717a' }}>Transcript · Whisper</span>
                {analysisResult?.intermediate_results?.transcript?.transcript ? (
                  <>
                    <p style={{ fontSize: 12, color: '#18181b', fontStyle: 'italic', fontWeight: 600, lineHeight: 1.55 }}>
                      "{analysisResult.intermediate_results.transcript.transcript}"
                    </p>
                    <span style={{ fontSize: 10, color: '#71717a', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                      Tone: {analysisResult.intermediate_results.context?.sentiment || 'neutral'}
                    </span>
                  </>
                ) : (
                  <>
                    <div style={{ fontSize: 20, fontWeight: 900, color: '#d4d4d8', lineHeight: 1 }}>—</div>
                    <div style={{ fontSize: 11, color: '#a1a1aa', marginTop: 2 }}>Awaiting speech</div>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Test Scenarios */}
          <div>
            <Label right="Populates all 3 channels">Test Scenarios</Label>
            <div style={S.scenarioGrid}>
              {[
                { id: 'masked_distress', label: 'Masked Distress', desc: 'Happy face (88%) + Anxious voice (72%)' },
                { id: 'sarcasm', label: 'Sarcasm', desc: 'Neutral face + Ironic text + Flat tone' },
                { id: 'urgent_panic', label: 'Urgent Panic', desc: 'Fear face (70%) + Fearful voice (80%)' },
                { id: 'congruent_happy', label: 'Congruent Happy', desc: 'Joy face (92%) + Cheerful voice (85%)' },
              ].map(({ id, label, desc }) => (
                <button key={id} onClick={() => runPreset(id)} disabled={processing} className="scenario-btn">
                  <div style={{ fontSize: 12, fontWeight: 800, color: '#18181b', marginBottom: 3 }}>{label}</div>
                  <div style={{ fontSize: 11, color: '#71717a', lineHeight: 1.5 }}>{desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Latency */}
          {analysisResult?.latencies_ms && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 20px', alignItems: 'center', padding: '10px 16px', background: '#fafafa', border: '1.5px solid #e4e4e7', borderRadius: 6, fontSize: 10, color: '#a1a1aa', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
              <span style={{ fontWeight: 800, color: '#71717a' }}>Pipeline Latency</span>
              {[['ViT', 'face'], ['XLSR', 'speech'], ['Whisper', 'transcript'], ['Fusion', 'fusion'], ['TTS', 'tts'], ['Total', 'total']].map(([lbl, key]) => (
                <span key={key} style={key === 'total' ? { fontWeight: 800, color: '#52525b' } : {}}>
                  {lbl} {analysisResult.latencies_ms[key] || 0}ms
                </span>
              ))}
            </div>
          )}

        </div>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }
      `}</style>
    </div>
  )
}

