import { useState } from 'react'

const DIRECTION_ARROW = {
  right: '→', left: '←', up: '↑', down: '↓',
  'up-right': '↗', 'up-left': '↖', 'down-right': '↘', 'down-left': '↙',
  slightly: '↔',
}

export default function AnalysisResult({ result }) {
  const [tab, setTab] = useState('comparison')
  const { stats, changes, caption, caption_mode } = result
  const imgSrc = tab === 'comparison' ? result.result_url : result.mask_url

  return (
    <div>
      {/* Stats row */}
      <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(5, 1fr)' }}>
        <div className="metric-card">
          <div className="metric-value color-indigo">{stats.objects_before}</div>
          <div className="metric-label">Objects Before</div>
        </div>
        <div className="metric-card">
          <div className="metric-value color-indigo">{stats.objects_after}</div>
          <div className="metric-label">Objects After</div>
        </div>
        <div className="metric-card">
          <div className="metric-value color-green">{stats.added}</div>
          <div className="metric-label">Added</div>
        </div>
        <div className="metric-card">
          <div className="metric-value color-red">{stats.removed}</div>
          <div className="metric-label">Removed</div>
        </div>
        <div className="metric-card">
          <div className="metric-value color-yellow">{stats.moved}</div>
          <div className="metric-label">Moved</div>
        </div>
      </div>

      {/* Change tables */}
      <div className="defects-grid" style={{ marginBottom: 24 }}>
        {/* Added */}
        <div className="defect-group">
          <div className="defect-group-title title-new">Added ({changes.added.length})</div>
          {changes.added.length === 0
            ? <p className="empty-state">None</p>
            : changes.added.map((c, i) => (
                <div className="defect-item" key={i}>
                  <span style={{ marginRight: 6 }}>+</span>{c.object}
                </div>
              ))}
        </div>

        {/* Removed */}
        <div className="defect-group">
          <div className="defect-group-title title-resolved">Removed ({changes.removed.length})</div>
          {changes.removed.length === 0
            ? <p className="empty-state">None</p>
            : changes.removed.map((c, i) => (
                <div className="defect-item" key={i}>
                  <span style={{ marginRight: 6 }}>−</span>{c.object}
                </div>
              ))}
        </div>

        {/* Moved */}
        <div className="defect-group">
          <div className="defect-group-title title-persist">Moved ({changes.moved.length})</div>
          {changes.moved.length === 0
            ? <p className="empty-state">None</p>
            : changes.moved.map((c, i) => (
                <div className="defect-item" key={i}>
                  {DIRECTION_ARROW[c.direction] || '→'} {c.object}
                  <span style={{ color: '#475569', fontSize: 11, marginLeft: 6 }}>
                    ({Math.round(c.distance)}px {c.direction})
                  </span>
                </div>
              ))}
        </div>
      </div>

      {/* Caption */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
          <span className="card-title" style={{ margin: 0 }}>Caption</span>
          <span className={`caption-badge ${caption_mode === 'vllm' ? 'badge-vllm' : 'badge-pattern'}`}>
            {caption_mode === 'vllm' ? '🤖 VLLM (Claude Vision)' : '⚙️ Pattern-Based'}
          </span>
        </div>
        <div className="summary-banner">{caption}</div>
      </div>

      {/* Image viewer */}
      <div className="image-tabs">
        <button className={`tab-btn ${tab === 'comparison' ? 'active' : ''}`} onClick={() => setTab('comparison')}>
          Side-by-Side
        </button>
        <button className={`tab-btn ${tab === 'mask' ? 'active' : ''}`} onClick={() => setTab('mask')}>
          Change Mask
        </button>
      </div>
      <div className="result-img-wrap">
        <img src={imgSrc} alt={tab} />
      </div>
    </div>
  )
}
