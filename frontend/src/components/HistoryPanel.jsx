import { useEffect, useState } from 'react'

function fmt(iso) {
  return new Date(iso).toLocaleString(undefined, {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

export default function HistoryPanel({ onSelect, refresh }) {
  const [items, setItems] = useState([])

  useEffect(() => {
    fetch('/api/analyses')
      .then(r => r.json())
      .then(setItems)
      .catch(() => {})
  }, [refresh])

  if (items.length === 0) return null

  return (
    <div className="card">
      <div className="card-title">Past Analyses</div>
      {items.map(item => (
        <div className="history-item" key={item.id} onClick={() => onSelect(item.id)}>
          <img
            className="history-thumb"
            src={item.result_url}
            alt="result"
            onError={(e) => { e.target.style.display = 'none' }}
          />
          <div className="history-meta">
            <div className="history-date">{fmt(item.created_at)}</div>
            <div className="history-summary">{item.caption}</div>
          </div>
          <div style={{ textAlign: 'center', flexShrink: 0 }}>
            <span className={`history-badge ${item.caption_mode === 'vllm' ? 'badge-vllm-sm' : 'badge-pattern-sm'}`}>
              {item.caption_mode === 'vllm' ? 'VLLM' : 'Pattern'}
            </span>
            <div style={{ fontSize: 11, color: '#475569', marginTop: 6 }}>
              +{item.added_count} −{item.removed_count} ↔{item.moved_count}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
