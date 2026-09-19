import React, { useState, useRef, useEffect } from 'react'
import './index.css'


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

  const displayEmotion = analysisResult?.primary_emotion || liveFaceReading?.emotion || 'Standby'

  return (
    <div className="min-h-screen bg-[#09090b] text-zinc-100 flex flex-col items-center py-8 px-4 antialiased">
      {/* Hidden audio element for TTS playback */}
      <audio ref={audioPlayerRef} />

      {/* Main Container */}
      <main className="w-full max-w-5xl flex flex-col gap-6">

        {/* Header */}
        <header className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-zinc-800/80 pb-5">
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white uppercase font-mono">
              EmpathAI
            </h1>
            <p className="text-xs text-zinc-400 mt-1">
              Multimodal emotion recognition & assistive speech synthesis
            </p>
          </div>

          {/* Mode Switcher */}
          <div className="flex items-center gap-1 bg-[#121215] p-1 rounded-md border border-zinc-800 text-xs font-mono">
            <span className="text-zinc-500 px-2 text-[11px] uppercase tracking-wider">Engine:</span>
            <button
              onClick={() => setFusionMode('fast')}
              className={`px-3 py-1 rounded text-xs transition-colors ${
                fusionMode === 'fast'
                  ? 'bg-white text-black font-semibold shadow-sm'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              Edge MLP
            </button>
            <button
              onClick={() => setFusionMode('full')}
              className={`px-3 py-1 rounded text-xs transition-colors ${
                fusionMode === 'full'
                  ? 'bg-white text-black font-semibold shadow-sm'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              Gemini
            </button>
          </div>
        </header>

        {/* Error message */}
        {errorMsg && (
          <div className="p-3.5 rounded-md bg-zinc-900 border border-zinc-700 text-zinc-200 text-xs flex items-center justify-between font-mono">
            <span>{errorMsg}</span>
            <button onClick={() => setErrorMsg(null)} className="text-zinc-400 hover:text-white underline ml-4">Dismiss</button>
          </div>
        )}

        {/* Workspace Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

          {/* Left Column: Camera / Capture Card (5 cols) */}
          <div className="lg:col-span-5 flex flex-col gap-4">
            <div className="bg-[#121215] border border-zinc-800 rounded-lg p-4 flex flex-col gap-3.5">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-mono uppercase tracking-widest text-zinc-400 font-medium">
                  Live Feed
                </span>
                {streamActive && (
                  <span className="flex items-center gap-1.5 text-[11px] font-mono text-zinc-300">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    ONLINE
                  </span>
                )}
              </div>

              {/* Video container */}
              <div className="relative aspect-video rounded-md bg-black border border-zinc-800/80 overflow-hidden flex items-center justify-center">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className={`w-full h-full object-cover ${streamActive ? 'block' : 'hidden'}`}
                />
                {!streamActive && (
                  <div className="text-center p-6 flex flex-col items-center gap-3">
                    <div className="w-9 h-9 rounded-full border border-zinc-800 bg-zinc-900 flex items-center justify-center text-zinc-400">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <p className="text-xs text-zinc-400 font-mono">Video source inactive</p>
                    <button
                      onClick={startCamera}
                      className="px-3.5 py-1.5 rounded-md bg-white hover:bg-zinc-200 text-black text-xs font-semibold font-mono transition-colors shadow-sm"
                    >
                      Start Camera & Mic
                    </button>
                  </div>
                )}

                {/* Recording indicator */}
                {recording && (
                  <div className="absolute top-2.5 left-2.5 bg-black/90 backdrop-blur-sm text-white px-2.5 py-1 rounded border border-red-500 text-xs font-mono font-semibold flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                    REC {recordSeconds}s
                  </div>
                )}

                {/* Processing indicator */}
                {processing && (
                  <div className="absolute inset-0 bg-black/85 backdrop-blur-xs flex flex-col items-center justify-center gap-3 text-center p-4">
                    <div className="w-6 h-6 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    <p className="text-xs font-mono text-zinc-300">
                      Processing multimodal fusion...
                    </p>
                  </div>
                )}

                {/* Minimalist Live HUD Badge */}
                {streamActive && showLiveHUD && liveFaceReading && !processing && (
                  <div className="absolute top-2.5 right-2.5 bg-black/85 backdrop-blur-md px-2.5 py-1 rounded border border-zinc-700/80 flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    <span className="text-xs font-mono font-medium text-white capitalize">
                      {liveFaceReading.emotion}
                    </span>
                    <span className="text-[11px] font-mono text-zinc-400">
                      {Math.round((liveFaceReading.confidence || 0) * 100)}%
                    </span>
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="flex flex-col gap-2 pt-1">
                {streamActive ? (
                  <div className="flex items-center gap-2">
                    {!recording ? (
                      <button
                        onClick={startRecording}
                        disabled={processing}
                        className="flex-1 py-2.5 px-3 rounded-md bg-white hover:bg-zinc-200 text-black font-semibold text-xs font-mono transition-colors flex items-center justify-center gap-2 disabled:opacity-40 shadow-sm"
                      >
                        <span className="w-2 h-2 rounded-full bg-red-600" />
                        Record Sample (Space)
                      </button>
                    ) : (
                      <button
                        onClick={stopRecording}
                        className="flex-1 py-2.5 px-3 rounded-md bg-red-600 hover:bg-red-500 text-white font-semibold text-xs font-mono transition-colors flex items-center justify-center gap-2"
                      >
                        <span className="w-2 h-2 bg-white rounded-xs" />
                        Stop Recording & Analyze
                      </button>
                    )}
                    <button
                      onClick={stopCamera}
                      className="px-3 py-2.5 rounded-md bg-zinc-900 hover:bg-zinc-800 text-zinc-300 text-xs font-mono border border-zinc-800 transition-colors"
                    >
                      Stop
                    </button>
                    <button
                      onClick={() => setShowLiveHUD(!showLiveHUD)}
                      className={`px-3 py-2.5 rounded-md text-xs font-mono border transition-colors ${
                        showLiveHUD
                          ? 'bg-zinc-800 text-white border-zinc-700'
                          : 'bg-zinc-950 text-zinc-500 border-zinc-800 hover:text-zinc-300'
                      }`}
                    >
                      HUD: {showLiveHUD ? 'ON' : 'OFF'}
                    </button>
                  </div>
                ) : null}

                {/* Minimalist Live Face Spectrum */}
                {streamActive && showLiveHUD && liveFaceReading?.all_scores && (
                  <div className="mt-1 p-3 rounded-md bg-black/40 border border-zinc-800/80 flex flex-col gap-2">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-mono uppercase tracking-widest text-zinc-400">
                        Live Face Spectrum
                      </span>
                      <span className="text-[10px] text-zinc-500 font-mono">ViT FP16</span>
                    </div>
                    <div className="flex flex-col gap-1.5">
                      {Object.entries(liveFaceReading.all_scores).map(([emo, score]) => {
                        const pct = Math.round(score * 100)
                        const isTop = emo === liveFaceReading.emotion
                        return (
                          <div key={emo} className="flex items-center gap-2 text-xs font-mono">
                            <span className={`w-18 capitalize text-[11px] ${isTop ? 'text-white font-semibold' : 'text-zinc-400'}`}>
                              {emo}
                            </span>
                            <div className="flex-1 h-1.5 bg-zinc-800/80 rounded-full overflow-hidden">
                              <div
                                className={`h-full transition-all duration-200 rounded-full ${isTop ? 'bg-white' : 'bg-zinc-500'}`}
                                style={{ width: `${pct}%` }}
                              />
                            </div>
                            <span className={`w-7 text-right text-[10px] ${isTop ? 'text-white font-semibold' : 'text-zinc-500'}`}>
                              {pct}%
                            </span>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}

                {/* Sample Scenarios */}
                <div className="mt-2 pt-3 border-t border-zinc-800/80">
                  <span className="text-[10px] font-mono uppercase tracking-widest text-zinc-500 block mb-2 font-medium">
                    Sample Scenarios
                  </span>
                  <div className="grid grid-cols-2 gap-2 font-mono">
                    <button
                      onClick={() => runPresetDemo('masked_distress')}
                      disabled={processing}
                      className="text-left p-2.5 rounded-md bg-zinc-900/40 hover:bg-zinc-900 border border-zinc-800/80 text-xs transition-colors group"
                    >
                      <div className="font-medium text-zinc-200 group-hover:text-white">Masked Distress</div>
                      <div className="text-[10px] text-zinc-500 mt-0.5">Smile + Anxious Voice</div>
                    </button>
                    <button
                      onClick={() => runPresetDemo('sarcasm')}
                      disabled={processing}
                      className="text-left p-2.5 rounded-md bg-zinc-900/40 hover:bg-zinc-900 border border-zinc-800/80 text-xs transition-colors group"
                    >
                      <div className="font-medium text-zinc-200 group-hover:text-white">Sarcasm</div>
                      <div className="text-[10px] text-zinc-500 mt-0.5">Positive Words + Flat Tone</div>
                    </button>
                    <button
                      onClick={() => runPresetDemo('urgent_panic')}
                      disabled={processing}
                      className="text-left p-2.5 rounded-md bg-zinc-900/40 hover:bg-zinc-900 border border-zinc-800/80 text-xs transition-colors group"
                    >
                      <div className="font-medium text-zinc-200 group-hover:text-white">Urgent Panic</div>
                      <div className="text-[10px] text-zinc-500 mt-0.5">Urgency Override</div>
                    </button>
                    <button
                      onClick={() => runPresetDemo('congruent_happy')}
                      disabled={processing}
                      className="text-left p-2.5 rounded-md bg-zinc-900/40 hover:bg-zinc-900 border border-zinc-800/80 text-xs transition-colors group"
                    >
                      <div className="font-medium text-zinc-200 group-hover:text-white">Congruent Happy</div>
                      <div className="text-[10px] text-zinc-500 mt-0.5">Smile + Warm Voice</div>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Fused Output & Reasoning (7 cols) */}
          <div className="lg:col-span-7 flex flex-col gap-4">

            {/* Mismatch Alert Banner */}
            {analysisResult?.mismatch_detected && (
              <div className="p-3.5 rounded-md bg-[#121215] border border-amber-500/50 flex items-start gap-3">
                <div className="w-1 self-stretch bg-amber-400 rounded-full" />
                <div className="flex-1 font-mono">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-white uppercase tracking-wider">
                      Channel Contradiction Detected
                    </span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300 border border-zinc-700">
                      Kind: {analysisResult.mismatch_kind}
                    </span>
                  </div>
                  <p className="text-xs text-zinc-400 mt-1 font-sans">
                    Nonverbal cues conflict across channels (facial expression vs. vocal cadence or spoken text).
                  </p>
                </div>
              </div>
            )}

            {/* Primary Emotion Insight Card */}
            <div className="rounded-lg p-5 border border-zinc-800 bg-[#121215] flex flex-col gap-4">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="uppercase tracking-widest text-zinc-500 text-[10px] font-semibold">
                  Primary Emotional Insight
                </span>
                {analysisResult ? (
                  <span className="text-zinc-400">
                    Confidence: {Math.round((analysisResult.confidence || 0) * 100)}%
                  </span>
                ) : liveFaceReading ? (
                  <span className="text-zinc-400 flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    Tracking Live Face
                  </span>
                ) : null}
              </div>

              <div>
                <div className="flex items-baseline gap-3">
                  <h2 className="text-4xl font-bold capitalize tracking-tight text-white font-sans">
                    {displayEmotion}
                  </h2>
                  {analysisResult?.urgency && analysisResult.urgency !== 'low' && (
                    <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-zinc-800 text-white border border-zinc-700 uppercase">
                      Urgency: {analysisResult.urgency}
                    </span>
                  )}
                </div>
                <p className="text-xs text-zinc-400 mt-1">
                  {analysisResult
                    ? (analysisResult.intermediate_results?.fusion?.source === 'mlp'
                        ? 'Synthesized via Edge MLP Classifier'
                        : 'Synthesized via Multimodal Fusion')
                    : (streamActive
                        ? 'Facial cues tracked live. Record utterance to fuse with voice and transcript.'
                        : 'Start camera and record an utterance to begin analysis.')}
                </p>
              </div>

              {/* Spoken Narration Block */}
              <div className="p-3.5 rounded-md bg-zinc-900/60 border border-zinc-800 flex flex-col gap-2">
                <div className="flex items-center justify-between text-xs font-mono text-zinc-400">
                  <span className="text-[10px] uppercase tracking-widest text-zinc-500 font-semibold">
                    Spoken Audio Feedback
                  </span>
                  {analysisResult?.audio_base64 && (
                    <button
                      onClick={() => {
                        if (audioPlayerRef.current) {
                          audioPlayerRef.current.currentTime = 0
                          audioPlayerRef.current.play()
                        }
                      }}
                      className="px-2.5 py-0.5 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs transition-colors"
                    >
                      Replay Audio
                    </button>
                  )}
                </div>
                <p className="text-sm font-medium text-zinc-100 font-sans leading-relaxed">
                  "{analysisResult?.spoken_summary || 'Record audio & video to synthesize spoken nonverbal insight.'}"
                </p>
              </div>
            </div>

            {/* Intermediate 3-Channel Breakdown */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {/* 1. Face */}
              <div className="p-3.5 rounded-md bg-[#121215] border border-zinc-800 flex flex-col gap-1.5">
                <span className="text-[10px] font-mono uppercase tracking-widest text-zinc-500">
                  Face (ViT)
                </span>
                {analysisResult?.intermediate_results?.face ? (
                  <div>
                    <div className="text-sm font-semibold capitalize text-white font-sans">
                      {analysisResult.intermediate_results.face.emotion || 'Detected'}
                    </div>
                    <p className="text-[11px] font-mono text-zinc-400 mt-0.5">
                      {Math.round((analysisResult.intermediate_results.face.confidence || 0) * 100)}% conf
                    </p>
                  </div>
                ) : liveFaceReading ? (
                  <div>
                    <div className="text-sm font-semibold capitalize text-white flex items-center gap-1.5 font-sans">
                      {liveFaceReading.emotion}
                      <span className="text-[9px] font-mono px-1 rounded bg-zinc-800 text-zinc-400 border border-zinc-700">LIVE</span>
                    </div>
                    <p className="text-[11px] font-mono text-zinc-400 mt-0.5">
                      {Math.round((liveFaceReading.confidence || 0) * 100)}% conf
                    </p>
                  </div>
                ) : (
                  <p className="text-xs text-zinc-500 font-mono">No face data</p>
                )}
              </div>

              {/* 2. Voice */}
              <div className="p-3.5 rounded-md bg-[#121215] border border-zinc-800 flex flex-col gap-1.5">
                <span className="text-[10px] font-mono uppercase tracking-widest text-zinc-500">
                  Voice (XLSR)
                </span>
                {analysisResult?.intermediate_results?.speech ? (
                  <div>
                    <div className="text-sm font-semibold capitalize text-white font-sans">
                      {analysisResult.intermediate_results.speech.emotion || 'Detected'}
                    </div>
                    <p className="text-[11px] font-mono text-zinc-400 mt-0.5">
                      {Math.round((analysisResult.intermediate_results.speech.confidence || 0) * 100)}% conf
                    </p>
                  </div>
                ) : (
                  <p className="text-xs text-zinc-500 font-mono">No audio data</p>
                )}
              </div>

              {/* 3. Transcript */}
              <div className="p-3.5 rounded-md bg-[#121215] border border-zinc-800 flex flex-col gap-1.5">
                <span className="text-[10px] font-mono uppercase tracking-widest text-zinc-500">
                  Transcript
                </span>
                {analysisResult?.intermediate_results?.transcript?.transcript ? (
                  <div>
                    <p className="text-xs text-zinc-200 line-clamp-2 font-sans">
                      "{analysisResult.intermediate_results.transcript.transcript}"
                    </p>
                    <span className="inline-block mt-1 text-[10px] font-mono text-zinc-400 uppercase">
                      Context: {analysisResult?.intermediate_results?.context?.sentiment || 'Neutral'}
                    </span>
                  </div>
                ) : (
                  <p className="text-xs text-zinc-500 font-mono">No transcript</p>
                )}
              </div>
            </div>

            {/* Latencies Telemetry */}
            {analysisResult?.latencies_ms && (
              <div className="p-3 rounded-md bg-[#121215] border border-zinc-800 text-[11px] font-mono text-zinc-400 flex flex-wrap items-center justify-between gap-2">
                <span className="text-zinc-500 uppercase tracking-widest text-[10px]">Latency Breakdown</span>
                <div className="flex items-center gap-3 text-[11px] flex-wrap">
                  <span>Face: {analysisResult.latencies_ms.face || 0}ms</span>
                  <span>Voice: {analysisResult.latencies_ms.speech || 0}ms</span>
                  <span>Whisper: {analysisResult.latencies_ms.transcript || 0}ms</span>
                  <span>Fusion: {analysisResult.latencies_ms.fusion || 0}ms</span>
                  <span>TTS: {analysisResult.latencies_ms.tts || 0}ms</span>
                  <span className="text-white font-semibold">Total: {analysisResult.latencies_ms.total || 0}ms</span>
                </div>
              </div>
            )}

          </div>
        </div>

      </main>
    </div>
  )
}
