import React, { useState, useRef, useEffect } from 'react'
import './index.css'

const EMOTION_COLORS = {
  happy: { bg: 'rgba(16, 185, 129, 0.15)', border: '#10b981', text: '#34d399', icon: '😊' },
  sad: { bg: 'rgba(59, 130, 246, 0.15)', border: '#3b82f6', text: '#60a5fa', icon: '😢' },
  angry: { bg: 'rgba(239, 68, 68, 0.15)', border: '#ef4444', text: '#f87171', icon: '😠' },
  fear: { bg: 'rgba(168, 85, 247, 0.15)', border: '#a855f7', text: '#c084fc', icon: '😨' },
  neutral: { bg: 'rgba(148, 163, 184, 0.15)', border: '#64748b', text: '#94a3b8', icon: '😐' },
  surprise: { bg: 'rgba(245, 158, 11, 0.15)', border: '#f59e0b', text: '#fbbf24', icon: '😲' },
  disgust: { bg: 'rgba(217, 119, 6, 0.15)', border: '#d97706', text: '#f59e0b', icon: '🤢' },
}

export default function App() {
  const [streamActive, setStreamActive] = useState(false)
  const [recording, setRecording] = useState(false)
  const [recordSeconds, setRecordSeconds] = useState(0)
  const [processing, setProcessing] = useState(false)
  const [errorMsg, setErrorMsg] = useState(null)
  const [analysisResult, setAnalysisResult] = useState(null)
  const [fusionMode, setFusionMode] = useState('full') // 'fast' | 'full'
  const [liveFaceReading, setLiveFaceReading] = useState(null)
  const [showLiveHUD, setShowLiveHUD] = useState(true)

  const videoRef = useRef(null)
  const mediaStreamRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])
  const audioPlayerRef = useRef(null)
  const timerRef = useRef(null)
  const bestFrameBlobRef = useRef(null)
  const bestFrameScoreRef = useRef(0)
  const recordingRef = useRef(false)
  recordingRef.current = recording

  // Initialize camera and mic
  const startCamera = async () => {
    try {
      setErrorMsg(null)
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: true,
      })
      mediaStreamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
      setStreamActive(true)
    } catch (err) {
      console.error('Camera/mic access error:', err)
      setErrorMsg('Could not access camera or microphone. Please allow media permissions.')
    }
  }

  // Stop camera and mic
  const stopCamera = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop())
      mediaStreamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setStreamActive(false)
    setLiveFaceReading(null)
  }

  // Start recording audio + snapshot
  const startRecording = () => {
    if (!mediaStreamRef.current) return
    audioChunksRef.current = []
    setRecordSeconds(0)
    setErrorMsg(null)
    bestFrameBlobRef.current = null
    bestFrameScoreRef.current = 0

    // Capture initial frame at the start of recording
    captureFrameBlob().then((b) => {
      if (b && !bestFrameBlobRef.current) {
        bestFrameBlobRef.current = b
        bestFrameScoreRef.current = 0.5
      }
    })

    try {
      const audioTrack = mediaStreamRef.current.getAudioTracks()[0]
      const audioStream = new MediaStream([audioTrack])
      const recorder = new MediaRecorder(audioStream, { mimeType: 'audio/webm' })

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          audioChunksRef.current.push(e.data)
        }
      }

      recorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        await submitAnalysis(audioBlob)
      }

      mediaRecorderRef.current = recorder
      recorder.start()
      setRecording(true)

      timerRef.current = setInterval(() => {
        setRecordSeconds((s) => s + 1)
      }, 1000)
    } catch (err) {
      console.error('MediaRecorder error:', err)
      setErrorMsg('Failed to record audio.')
    }
  }

  // Stop recording audio
  const stopRecording = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop()
    }
    setRecording(false)
  }

  // Grab current frame as JPEG blob
  const captureFrameBlob = async () => {
    if (!videoRef.current) return null
    const video = videoRef.current
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth || 640
    canvas.height = video.videoHeight || 480
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    return new Promise((resolve) => {
      canvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.85)
    })
  }

  // Submit to /analyze
  const submitAnalysis = async (audioBlob) => {
    setProcessing(true)
    setErrorMsg(null)

    try {
      // Use best expressive frame captured during the recording session, or capture current frame
      let frameBlob = bestFrameBlobRef.current
      if (!frameBlob) {
        frameBlob = await captureFrameBlob()
      }
      const formData = new FormData()

      if (frameBlob) {
        formData.append('image', frameBlob, 'webcam_frame.jpg')
      }
      if (audioBlob) {
        formData.append('audio', audioBlob, 'speech_audio.webm')
      }
      formData.append('mode', fusionMode)
      formData.append('generate_tts', 'true')

      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}))
        throw new Error(errJson.detail || `Server error: ${response.status}`)
      }

      const result = await response.json()
      setAnalysisResult(result)

      // Auto-play TTS audio if returned
      if (result.audio_base64 && audioPlayerRef.current) {
        audioPlayerRef.current.src = result.audio_base64
        audioPlayerRef.current.play().catch((e) => console.log('Autoplay blocked:', e))
      }
    } catch (err) {
      console.error('Analysis error:', err)
      setErrorMsg(`Analysis failed: ${err.message}`)
    } finally {
      setProcessing(false)
    }
  }

  // Sample scenario simulator
  const runPresetDemo = async (scenario) => {
    setProcessing(true)
    setErrorMsg(null)

    let payload = {}
    if (scenario === 'masked_distress') {
      payload = {
        face: { probs: { happy: 0.88, neutral: 0.08, sad: 0.04 }, confidence: 0.88 },
        speech: { probs: { fearful: 0.72, sad: 0.18, neutral: 0.10 }, confidence: 0.75 },
        context: { sentiment: 'positive', confidence: 0.85, transcript: "I'm completely fine, really.", urgency: 'low' },
        mode: fusionMode,
      }
    } else if (scenario === 'sarcasm') {
      payload = {
        face: { probs: { neutral: 0.60, disgust: 0.25, angry: 0.15 }, confidence: 0.60 },
        speech: { probs: { angry: 0.55, disgust: 0.25, neutral: 0.20 }, confidence: 0.65 },
        context: { sentiment: 'sarcastic', sarcasm_detected: true, confidence: 0.90, transcript: 'Oh brilliant, another flat tire. Just what I needed!', urgency: 'medium' },
        mode: fusionMode,
      }
    } else if (scenario === 'urgent_panic') {
      payload = {
        face: { probs: { fear: 0.70, surprise: 0.20, neutral: 0.10 }, confidence: 0.75 },
        speech: { probs: { fearful: 0.80, angry: 0.10, sad: 0.10 }, confidence: 0.85 },
        context: { sentiment: 'urgent', confidence: 0.95, transcript: 'Emergency! Call an ambulance immediately!', urgency: 'high' },
        mode: fusionMode,
      }
    } else {
      // Congruent happy
      payload = {
        face: { probs: { happy: 0.92, neutral: 0.05, surprise: 0.03 }, confidence: 0.92 },
        speech: { probs: { happy: 0.85, calm: 0.10, neutral: 0.05 }, confidence: 0.85 },
        context: { sentiment: 'positive', confidence: 0.90, transcript: 'We finished the project and everything looks amazing!', urgency: 'low' },
        mode: fusionMode,
      }
    }

    try {
      const fuseResp = await fetch('http://localhost:8000/fuse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const fuseData = await fuseResp.json()

      // Fetch speech audio
      const speakResp = await fetch('http://localhost:8000/speak', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: fuseData.narration,
          label: fuseData.label,
          mismatch: fuseData.mismatch,
          mismatch_kind: fuseData.mismatch_kind,
        }),
      })
      const audioBlob = await speakResp.blob()
      const audioUrl = URL.createObjectURL(audioBlob)

      setAnalysisResult({
        primary_emotion: fuseData.label,
        confidence: fuseData.confidence,
        mismatch_detected: fuseData.mismatch,
        mismatch_kind: fuseData.mismatch_kind,
        urgency: fuseData.urgency,
        spoken_summary: fuseData.narration,
        audio_base64: audioUrl,
        intermediate_results: {
          face: payload.face,
          speech: payload.speech,
          transcript: { transcript: payload.context.transcript },
          context: payload.context,
          fusion: fuseData,
        },
        latencies_ms: { face: 190.2, speech: 82.5, transcript: 210.0, context: 350.0, fusion: fuseData.latency_ms || 1.2, tts: 480.0, total: 1313.9 },
      })

      if (audioPlayerRef.current) {
        audioPlayerRef.current.src = audioUrl
        audioPlayerRef.current.play().catch(() => {})
      }
    } catch (err) {
      console.error('Demo simulation error:', err)
      setErrorMsg(`Preset demo failed: ${err.message}`)
    } finally {
      setProcessing(false)
    }
  }

  // Keyboard shortcut: Spacebar to Record/Stop
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
        e.preventDefault()
        if (recording) {
          stopRecording()
        } else if (streamActive) {
          startRecording()
        }
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [recording, streamActive])

  // Cleanup stream on unmount
  useEffect(() => {
    return () => {
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((t) => t.stop())
      }
    }
  }, [])

  // Live face inspection polling when camera is active (mirrors live_face_view.py)
  useEffect(() => {
    if (!streamActive || processing || !showLiveHUD) {
      return
    }

    let isSubscribed = true
    let isRequestBusy = false

    const pollLiveFace = async () => {
      if (isRequestBusy || !videoRef.current || videoRef.current.readyState < 2) return
      isRequestBusy = true

      try {
        const video = videoRef.current
        const canvas = document.createElement('canvas')
        canvas.width = Math.min(video.videoWidth || 640, 320)
        canvas.height = Math.min(video.videoHeight || 480, 240)
        const ctx = canvas.getContext('2d')
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

        const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.8))
        if (!blob || !isSubscribed) return

        const formData = new FormData()
        formData.append('image', blob, 'live_frame.jpg')

        const res = await fetch('http://localhost:8000/face-emotion', {
          method: 'POST',
          body: formData,
        })
        if (res.ok && isSubscribed) {
          const data = await res.json()
          setLiveFaceReading(data)

          // If currently recording, retain the most expressive frame
          if (recordingRef.current && blob) {
            const isNonNeutral = data.emotion && data.emotion.toLowerCase() !== 'neutral'
            const score = (data.confidence || 0) + (isNonNeutral ? 1.0 : 0.0)
            if (score > bestFrameScoreRef.current || !bestFrameBlobRef.current) {
              bestFrameBlobRef.current = blob
              bestFrameScoreRef.current = score
            }
          }
        }
      } catch (err) {
        // silent fail on transient network hiccups
      } finally {
        isRequestBusy = false
      }
    }

    const interval = setInterval(pollLiveFace, 650)
    return () => {
      isSubscribed = false
      clearInterval(interval)
    }
  }, [streamActive, processing, showLiveHUD])

  const currentEmotion = analysisResult?.primary_emotion?.toLowerCase() || 'neutral'
  const emotionStyle = EMOTION_COLORS[currentEmotion] || EMOTION_COLORS.neutral

  return (
    <div className="min-h-screen text-slate-100 flex flex-col items-center py-8 px-4 relative overflow-x-hidden"
         style={{ background: 'var(--color-bg)' }}>

      {/* Hidden audio element for TTS playback */}
      <audio ref={audioPlayerRef} />

      {/* Ambient background glow */}
      <div
        style={{
          position: 'fixed',
          top: '20%',
          left: '50%',
          transform: 'translateX(-50%)',
          width: '700px',
          height: '700px',
          borderRadius: '50%',
          background: 'var(--color-accent-glow)',
          filter: 'blur(160px)',
          pointerEvents: 'none',
          zIndex: 0,
        }}
      />

      {/* Main Container */}
      <main className="w-full max-w-5xl z-10 flex flex-col gap-6">

        {/* Header */}
        <header className="flex flex-col md:flex-row items-center justify-between gap-4 border-b border-white/10 pb-5">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-3xl font-extrabold tracking-tight">
                Empath<span className="text-violet-400">AI</span>
              </h1>
              <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                Phase 5 Active
              </span>
            </div>
            <p className="text-sm text-slate-400 mt-1">
              Multimodal emotion recognition & mismatch detection for visually impaired users.
            </p>
          </div>

          {/* Mode Switcher */}
          <div className="flex items-center gap-2 bg-slate-900/80 p-1.5 rounded-xl border border-white/10 text-xs font-medium">
            <span className="text-slate-400 px-2">Reasoning:</span>
            <button
              onClick={() => setFusionMode('fast')}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                fusionMode === 'fast'
                  ? 'bg-violet-600 text-white shadow-lg'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Fast Edge (&lt;2ms)
            </button>
            <button
              onClick={() => setFusionMode('full')}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                fusionMode === 'full'
                  ? 'bg-violet-600 text-white shadow-lg'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Gemini Synthesis
            </button>
          </div>
        </header>

        {/* Error message */}
        {errorMsg && (
          <div className="p-4 rounded-xl bg-red-500/15 border border-red-500/40 text-red-300 text-sm flex items-center justify-between">
            <span>⚠️ {errorMsg}</span>
            <button onClick={() => setErrorMsg(null)} className="text-red-400 hover:text-white text-xs underline">Dismiss</button>
          </div>
        )}

        {/* Live Camera + Controls Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

          {/* Left Column: Camera / Capture Card (5 cols) */}
          <div className="lg:col-span-5 flex flex-col gap-4">
            <div className="bg-slate-900/90 border border-white/10 rounded-2xl p-4 flex flex-col gap-3 shadow-xl">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono uppercase tracking-wider text-slate-400">
                  Video & Audio Feed
                </span>
                {streamActive && (
                  <span className="flex items-center gap-1.5 text-xs text-emerald-400">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                    Live
                  </span>
                )}
              </div>

              {/* Video container */}
              <div className="relative aspect-video rounded-xl bg-slate-950 border border-white/5 overflow-hidden flex items-center justify-center">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className={`w-full h-full object-cover ${streamActive ? 'block' : 'hidden'}`}
                />
                {!streamActive && (
                  <div className="text-center p-6 flex flex-col items-center gap-3">
                    <div className="w-12 h-12 rounded-full bg-white/5 flex items-center justify-center text-2xl">
                      📷
                    </div>
                    <p className="text-xs text-slate-400">Camera is currently stopped</p>
                    <button
                      onClick={startCamera}
                      className="px-4 py-2 rounded-xl bg-violet-600 hover:bg-violet-500 text-white text-xs font-semibold shadow-lg transition-all"
                    >
                      Enable Camera & Mic
                    </button>
                  </div>
                )}

                {/* Recording indicator overlay */}
                {recording && (
                  <div className="absolute top-3 left-3 bg-red-600/90 backdrop-blur-md text-white px-3 py-1 rounded-full text-xs font-bold flex items-center gap-2 shadow-lg animate-pulse">
                    <span className="w-2 h-2 rounded-full bg-white" />
                    REC {recordSeconds}s
                  </div>
                )}

                {/* Processing overlay */}
                {processing && (
                  <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm flex flex-col items-center justify-center gap-3 text-center p-4">
                    <div className="w-8 h-8 border-3 border-violet-500 border-t-transparent rounded-full animate-spin" />
                    <p className="text-sm font-medium text-violet-300">
                      Analyzing Face, Voice & Context...
                    </p>
                  </div>
                )}

                {/* Live Face Emotion HUD Overlay (mirrors live_face_view.py) */}
                {streamActive && showLiveHUD && liveFaceReading && !processing && (
                  <div className="absolute top-3 right-3 bg-slate-950/85 backdrop-blur-md px-3 py-1.5 rounded-xl border border-white/10 flex items-center gap-2 shadow-xl animate-fade-in">
                    <span className="text-base">
                      {EMOTION_COLORS[liveFaceReading.emotion?.toLowerCase()]?.icon || '😐'}
                    </span>
                    <div className="flex flex-col">
                      <span className="text-xs font-bold capitalize text-white leading-tight flex items-center gap-1.5">
                        {liveFaceReading.emotion}
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                      </span>
                      <span className="text-[10px] text-slate-400 font-mono">
                        {Math.round((liveFaceReading.confidence || 0) * 100)}% live
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="flex flex-col gap-2 pt-2">
                {streamActive ? (
                  <div className="flex items-center gap-2">
                    {!recording ? (
                      <button
                        onClick={startRecording}
                        disabled={processing}
                        className="flex-1 py-3 px-4 rounded-xl bg-red-600 hover:bg-red-500 text-white font-bold text-sm shadow-lg flex items-center justify-center gap-2 transition-all disabled:opacity-50"
                      >
                        <span className="w-3 h-3 rounded-full bg-white" />
                        Record Emotion Sample (Space)
                      </button>
                    ) : (
                      <button
                        onClick={stopRecording}
                        className="flex-1 py-3 px-4 rounded-xl bg-amber-500 hover:bg-amber-400 text-slate-950 font-extrabold text-sm shadow-lg flex items-center justify-center gap-2 animate-pulse transition-all"
                      >
                        <span className="w-3 h-3 bg-slate-950 rounded-sm" />
                        Stop & Analyze Sample
                      </button>
                    )}
                    <button
                      onClick={stopCamera}
                      className="px-3 py-3 rounded-xl bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white text-xs border border-white/10"
                      title="Stop Camera"
                    >
                      Turn Off
                    </button>
                    {showLiveHUD ? (
                      <button
                        onClick={() => setShowLiveHUD(false)}
                        className="px-2.5 py-3 rounded-xl bg-cyan-500/15 border border-cyan-500/30 text-cyan-300 text-xs hover:bg-cyan-500/25 transition-all"
                        title="Hide Live Face HUD"
                      >
                        👁️ HUD: ON
                      </button>
                    ) : (
                      <button
                        onClick={() => setShowLiveHUD(true)}
                        className="px-2.5 py-3 rounded-xl bg-white/5 border border-white/10 text-slate-400 text-xs hover:text-white transition-all"
                        title="Show Live Face HUD"
                      >
                        👁️ HUD: OFF
                      </button>
                    )}
                  </div>
                ) : null}

                {/* Live Face Emotion Spectrum Panel (from live_face_view.py) */}
                {streamActive && showLiveHUD && liveFaceReading?.all_scores && (
                  <div className="mt-2 p-3 rounded-xl bg-slate-950/60 border border-white/5 flex flex-col gap-2">
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                        Live Face Spectrum (ViT FP16 on GPU)
                      </span>
                      <span className="text-[10px] text-emerald-400 font-mono">
                        Active
                      </span>
                    </div>
                    <div className="grid grid-cols-1 gap-1">
                      {Object.entries(liveFaceReading.all_scores).map(([emo, score]) => {
                        const pct = Math.round(score * 100)
                        const isTop = emo === liveFaceReading.emotion
                        const style = EMOTION_COLORS[emo] || EMOTION_COLORS.neutral
                        return (
                          <div key={emo} className="flex items-center gap-2 text-xs font-mono">
                            <span className="w-20 capitalize text-[11px] text-slate-400 flex items-center gap-1">
                              <span>{style.icon}</span>
                              <span className={isTop ? 'text-white font-semibold' : ''}>{emo}</span>
                            </span>
                            <div className="flex-1 h-2 bg-slate-800/80 rounded-full overflow-hidden relative">
                              <div
                                className="h-full transition-all duration-300 rounded-full"
                                style={{
                                  width: `${pct}%`,
                                  backgroundColor: isTop ? (style.border || '#10b981') : 'rgba(148, 163, 184, 0.4)',
                                }}
                              />
                            </div>
                            <span className={`w-8 text-right text-[10px] ${isTop ? 'text-white font-bold' : 'text-slate-500'}`}>
                              {pct}%
                            </span>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}

                {/* Quick Preset Demos */}
                <div className="mt-2 pt-3 border-t border-white/5">
                  <p className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-2">
                    Or Test Interactive Presets:
                  </p>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      onClick={() => runPresetDemo('masked_distress')}
                      disabled={processing}
                      className="text-left px-2.5 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/5 text-xs text-slate-300 transition-all"
                    >
                      🎭 <span className="font-semibold">Masked Distress</span>
                      <p className="text-[10px] text-slate-500">Smile + Shaky fear voice</p>
                    </button>
                    <button
                      onClick={() => runPresetDemo('sarcasm')}
                      disabled={processing}
                      className="text-left px-2.5 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/5 text-xs text-slate-300 transition-all"
                    >
                      😏 <span className="font-semibold">Sarcasm</span>
                      <p className="text-[10px] text-slate-500">"Great, flat tire!"</p>
                    </button>
                    <button
                      onClick={() => runPresetDemo('urgent_panic')}
                      disabled={processing}
                      className="text-left px-2.5 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/5 text-xs text-slate-300 transition-all"
                    >
                      🚨 <span className="font-semibold">Urgent Alarm</span>
                      <p className="text-[10px] text-slate-500">Urgency override</p>
                    </button>
                    <button
                      onClick={() => runPresetDemo('congruent_happy')}
                      disabled={processing}
                      className="text-left px-2.5 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/5 text-xs text-slate-300 transition-all"
                    >
                      ✨ <span className="font-semibold">Congruent Happy</span>
                      <p className="text-[10px] text-slate-500">Smile + Warm voice</p>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Live Fused Output & Reasoning (7 cols) */}
          <div className="lg:col-span-7 flex flex-col gap-4">

            {/* Mismatch Alert Banner (Only when mismatch is detected) */}
            {analysisResult?.mismatch_detected && (
              <div className="p-4 rounded-2xl bg-amber-500/15 border-2 border-amber-500/60 shadow-lg shadow-amber-500/10 flex items-start gap-3.5 animate-fadeIn">
                <span className="text-2xl mt-0.5">⚠️</span>
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <h3 className="font-bold text-amber-300 text-sm uppercase tracking-wide">
                      Channel Contradiction Detected
                    </h3>
                    <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-amber-500/20 text-amber-300 border border-amber-500/40">
                      Kind: {analysisResult.mismatch_kind}
                    </span>
                  </div>
                  <p className="text-xs text-amber-200/90 mt-1 leading-relaxed">
                    The speaker's nonverbal signals contradict each other (e.g. facial expression vs. vocal cadence or words).
                  </p>
                </div>
              </div>
            )}

            {/* Fused Emotion Main Card */}
            <div
              className="rounded-2xl p-6 border shadow-xl flex flex-col gap-4 relative overflow-hidden transition-all"
              style={{
                backgroundColor: emotionStyle.bg,
                borderColor: emotionStyle.border,
              }}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono uppercase tracking-wider text-slate-400">
                  Primary Emotional Insight
                </span>
                {analysisResult && (
                  <span className="text-xs font-mono text-slate-400">
                    Confidence: {Math.round((analysisResult.confidence || 0) * 100)}%
                  </span>
                )}
              </div>

              <div className="flex items-center gap-4">
                <div
                  className="w-16 h-16 rounded-2xl flex items-center justify-center text-4xl shadow-inner"
                  style={{ background: 'rgba(0,0,0,0.3)', border: `1px solid ${emotionStyle.border}` }}
                >
                  {emotionStyle.icon}
                </div>
                <div>
                  <div className="flex items-center gap-3">
                    <h2 className="text-3xl font-black capitalize tracking-tight" style={{ color: emotionStyle.text }}>
                      {analysisResult?.primary_emotion || 'Waiting for input'}
                    </h2>
                    {analysisResult?.urgency && analysisResult.urgency !== 'low' && (
                      <span className="px-2 py-0.5 rounded-full text-xs font-bold bg-red-500/20 text-red-400 border border-red-500/40">
                        Urgency: {analysisResult.urgency.toUpperCase()}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-slate-400 mt-1">
                    {analysisResult?.intermediate_results?.fusion?.source === 'mlp'
                      ? 'Decided by 3-Layer Edge MLP (<1ms)'
                      : 'Synthesized via Multimodal Logic'}
                  </p>
                </div>
              </div>

              {/* Spoken Summary & Audio Player */}
              <div className="mt-2 p-4 rounded-xl bg-slate-950/60 border border-white/10 flex flex-col gap-2.5">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold flex items-center gap-1.5">
                    🔊 Spoken Narration (Assistive Audio)
                  </span>
                  {analysisResult?.audio_base64 && (
                    <button
                      onClick={() => {
                        if (audioPlayerRef.current) {
                          audioPlayerRef.current.currentTime = 0
                          audioPlayerRef.current.play()
                        }
                      }}
                      className="px-2.5 py-1 rounded-md bg-white/10 hover:bg-white/20 text-white text-xs font-medium transition-all"
                    >
                      Replay Voice ↺
                    </button>
                  )}
                </div>
                <p className="text-sm font-medium text-slate-200 leading-relaxed italic">
                  "{analysisResult?.spoken_summary || 'Record audio & video to receive spoken emotional feedback.'}"
                </p>
              </div>
            </div>

            {/* Intermediate 3-Channel Breakdown */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">

              {/* 1. Face */}
              <div className="p-3.5 rounded-xl bg-slate-900/80 border border-white/10 flex flex-col gap-2">
                <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  👤 Face ViT + FACS
                </span>
                {analysisResult?.intermediate_results?.face ? (
                  <div>
                    <div className="text-sm font-bold capitalize text-violet-300">
                      {analysisResult.intermediate_results.face.emotion || 'Detected'}
                    </div>
                    <p className="text-[11px] text-slate-400">
                      Conf: {Math.round((analysisResult.intermediate_results.face.confidence || 0) * 100)}%
                    </p>
                  </div>
                ) : (
                  <p className="text-xs text-slate-500 italic">No face data</p>
                )}
              </div>

              {/* 2. Voice */}
              <div className="p-3.5 rounded-xl bg-slate-900/80 border border-white/10 flex flex-col gap-2">
                <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  🎙️ Vocal Tone (XLSR)
                </span>
                {analysisResult?.intermediate_results?.speech ? (
                  <div>
                    <div className="text-sm font-bold capitalize text-blue-300">
                      {analysisResult.intermediate_results.speech.emotion || 'Detected'}
                    </div>
                    <p className="text-[11px] text-slate-400">
                      Conf: {Math.round((analysisResult.intermediate_results.speech.confidence || 0) * 100)}%
                    </p>
                  </div>
                ) : (
                  <p className="text-xs text-slate-500 italic">No audio data</p>
                )}
              </div>

              {/* 3. Spoken Context */}
              <div className="p-3.5 rounded-xl bg-slate-900/80 border border-white/10 flex flex-col gap-2">
                <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  💬 Spoken Words
                </span>
                {analysisResult?.intermediate_results?.transcript?.transcript ? (
                  <div>
                    <p className="text-xs text-slate-300 line-clamp-2 italic">
                      "{analysisResult.intermediate_results.transcript.transcript}"
                    </p>
                    <span className="inline-block mt-1 px-1.5 py-0.5 rounded text-[10px] font-semibold bg-white/10 text-slate-300">
                      {analysisResult?.intermediate_results?.context?.sentiment || 'contextual'}
                    </span>
                  </div>
                ) : (
                  <p className="text-xs text-slate-500 italic">No transcript</p>
                )}
              </div>
            </div>

            {/* Latency Breakdown Pill Bar */}
            {analysisResult?.latencies_ms && (
              <div className="p-3 rounded-xl bg-slate-950/40 border border-white/5 flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-slate-400">
                <span>⏱️ Stage Latencies:</span>
                <span className="bg-white/5 px-2 py-0.5 rounded">Face: {analysisResult.latencies_ms.face || 0}ms</span>
                <span className="bg-white/5 px-2 py-0.5 rounded">Voice: {analysisResult.latencies_ms.speech || 0}ms</span>
                <span className="bg-white/5 px-2 py-0.5 rounded">Whisper: {analysisResult.latencies_ms.transcript || 0}ms</span>
                <span className="bg-white/5 px-2 py-0.5 rounded">Context: {analysisResult.latencies_ms.context || 0}ms</span>
                <span className="bg-violet-500/20 text-violet-300 px-2 py-0.5 rounded">Fusion: {analysisResult.latencies_ms.fusion || 0}ms</span>
                <span className="bg-white/5 px-2 py-0.5 rounded">TTS: {analysisResult.latencies_ms.tts || 0}ms</span>
                <span className="bg-emerald-500/20 text-emerald-300 font-bold px-2 py-0.5 rounded">
                  Total: {analysisResult.latencies_ms.total || 0}ms
                </span>
              </div>
            )}

          </div>
        </div>

      </main>
    </div>
  )
}
