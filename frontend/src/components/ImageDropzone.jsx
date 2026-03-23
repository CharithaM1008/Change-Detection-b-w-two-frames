import { useRef, useState } from 'react'

export default function ImageDropzone({ label, badgeClass, file, onFile }) {
  const inputRef = useRef()
  const [drag, setDrag] = useState(false)
  const preview = file ? URL.createObjectURL(file) : null

  const handle = (f) => { if (f && f.type.startsWith('image/')) onFile(f) }

  return (
    <div
      className={`dropzone ${drag ? 'drag-over' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); handle(e.dataTransfer.files[0]) }}
      onClick={() => inputRef.current.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={(e) => handle(e.target.files[0])}
        onClick={(e) => e.stopPropagation()}
      />
      <span className={`dropzone-badge ${badgeClass}`}>{label}</span>
      {preview ? (
        <>
          <img src={preview} alt="preview" className="dropzone-preview" />
          <p className="dropzone-name">{file.name}</p>
        </>
      ) : (
        <>
          <div className="dropzone-icon">🖼️</div>
          <p className="dropzone-label">
            <span>Click to upload</span> or drag & drop
          </p>
          <p className="dropzone-label" style={{ marginTop: 6, fontSize: 12 }}>PNG, JPG, JPEG</p>
        </>
      )}
    </div>
  )
}
