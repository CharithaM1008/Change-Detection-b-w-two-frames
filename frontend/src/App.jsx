import { useState } from 'react'
import ImageDropzone from './components/ImageDropzone'
import AnalysisResult from './components/AnalysisResult'
import HistoryPanel from './components/HistoryPanel'

export default function App() {
  const [before, setBefore] = useState(null)
  const [after, setAfter]   = useState(null)
  const [captionMode, setCaptionMode] = useState('pattern')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]   = useState(null)
  const [historyKey, setHistoryKey] = useState(0)

  const canAnalyze = before && after && !loading

  async function analyze() {
    setLoading(true)
    setError(null)
    setResult(null)

    const form = new FormData()
    form.append('before', before)
    form.append('after', after)
    form.append('caption_mode', captionMode)

    try {
      const res = await fetch('/api/analyze', { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Analysis failed')
      }
      const data = await res.json()
      setResult(data)
      setHistoryKey(k => k + 1)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function loadHistoryItem(id) {
    const res = await fetch(`/api/analyses/${id}`)
    const data = await res.json()
    setResult(data)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <>
      <header className="header">
        <span className="header-icon">📹</span>
        <div>
          <h1>Change Detection System</h1>
          <p>YOLO-powered object-level change detection between two frames</p>
        </div>
      </header>

      <main className="container">
        {/* Upload + Options */}
        <div className="card">
          <div className="card-title">Upload Frames</div>
          <div className="upload-grid">
            <ImageDropzone label="BEFORE" badgeClass="badge-before" file={before} onFile={setBefore} />
            <ImageDropzone label="AFTER"  badgeClass="badge-after"  file={after}  onFile={setAfter}  />
          </div>

          {/* Caption mode toggle */}
          <div className="caption-toggle-wrap">
            <span className="toggle-label">Caption Generation</span>
            <div className="toggle-group">
              <button
                className={`toggle-btn ${captionMode === 'pattern' ? 'active-pattern' : ''}`}
                onClick={() => setCaptionMode('pattern')}
              >
                Pattern-Based
              </button>
              <button
                className={`toggle-btn ${captionMode === 'vllm' ? 'active-vllm' : ''}`}
                onClick={() => setCaptionMode('vllm')}
              >
                VLLM (Claude Vision)
              </button>
            </div>
            <p className="toggle-hint">
              {captionMode === 'pattern'
                ? 'Rule-based: generates captions from detected object changes (no API needed).'
                : 'Vision LLM: sends frames to Claude Vision for a detailed natural language description. Requires ANTHROPIC_API_KEY.'}
            </p>
          </div>

          <button className="btn-analyze" onClick={analyze} disabled={!canAnalyze}>
            {loading ? 'Analyzing...' : 'Analyze Changes'}
          </button>
        </div>

        {/* Loader */}
        {loading && (
          <div className="card loader-wrap">
            <div className="spinner" />
            <p style={{ color: '#64748b', fontSize: 14 }}>
              {captionMode === 'vllm'
                ? 'Running YOLO detection + Claude Vision caption...'
                : 'Running YOLO detection...'}
            </p>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="card" style={{ borderColor: '#7f1d1d' }}>
            <p style={{ color: '#f87171', fontSize: 14 }}>Error: {error}</p>
          </div>
        )}

        {/* Results */}
        {result && !loading && (
          <div className="card">
            <div className="card-title">Analysis Results</div>
            <AnalysisResult result={result} />
          </div>
        )}

        {/* History */}
        <HistoryPanel refresh={historyKey} onSelect={loadHistoryItem} />
      </main>
    </>
  )
}
