# Change Detection Between Two Frames

An AI-powered system that detects, classifies, and explains object-level changes between two image frames — built for CCTV and security surveillance use cases.

---

## What It Does

Given a **BEFORE** and an **AFTER** image from the same camera, the system:

- Detects which objects were **added**, **removed**, or **moved**
- Computes a **change map** showing exactly where in the frame things changed
- Generates a **natural language caption** describing what happened — either rule-based or via Claude Vision (VLLM)
- Stores every analysis in a **SQLite database** with a full history view

---

## How It Works

```
BEFORE image ──┐
               ├──► Frame Alignment (ORB + Homography)
AFTER image  ──┘
                        │
                        ▼
              SSIM Change Map (localize changed regions)
                        │
                        ▼
              YOLOv8 Object Detection (both frames)
                        │
                        ▼
              Object Matching (class + spatial proximity)
                        │
                        ▼
         ┌──────────────┴──────────────┐
         ▼                             ▼
  Pattern Caption              VLLM Caption
  (rule-based from             (Claude Haiku Vision
   detected changes)            analyzes both frames)
```

**Frame Alignment** — ORB keypoints + RANSAC homography corrects minor camera jitter before any comparison.

**SSIM Change Map** — Structural Similarity Index highlights regions that structurally changed, ignoring minor noise.

**YOLOv8 Detection** — Pre-trained YOLOv8n detects 80 object classes (people, chairs, bags, vehicles, etc.) in both frames.

**Object Matching** — Objects matched by class label + centroid proximity. Unmatched objects in AFTER = added; unmatched in BEFORE = removed; matched but moved = moved.

**Caption Modes:**
- `Pattern` — Generates a description directly from the detected changes. No API key needed. Fast.
- `VLLM` — Sends both frames to Claude Haiku Vision for a detailed, context-aware description. Requires `ANTHROPIC_API_KEY`.

---

## Project Structure

```
├── change_detection_system.py   Core detection logic (YOLO + SSIM + matching)
├── wall_defect_detector.py      Wall surface defect module (texture/SSIM/YOLO)
├── integrated_system.py         Combined object + defect pipeline
│
├── backend/
│   ├── main.py                  FastAPI app (all API routes)
│   ├── database.py              SQLAlchemy + SQLite setup
│   ├── models.py                Analysis DB model
│   ├── requirements.txt         Python dependencies
│   ├── uploads/                 Stored input images (auto-created)
│   └── results/                 Stored output images (auto-created)
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx              Main layout + state
│   │   ├── App.css              Dark theme styles
│   │   └── components/
│   │       ├── ImageDropzone.jsx    Drag & drop upload
│   │       ├── AnalysisResult.jsx   Results display
│   │       └── HistoryPanel.jsx     Past analyses
│   ├── package.json
│   └── vite.config.js           Proxies /api → localhost:8000
│
├── start_backend.bat            One-click backend start
└── start_frontend.bat           One-click frontend start
```

---

## Prerequisites

- Python 3.9+
- Node.js 18+
- pip

---

## How to Run

### Step 1 — Clone / open the project

```bash
cd Change-Detection-b-w-two-frames
```

### Step 2 — Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

> On first run, YOLOv8 will automatically download `yolov8n.pt` (~6 MB). This happens once.

### Step 3 — (Optional) Set API key for VLLM mode

Only needed if you want to use **VLLM (Claude Vision)** caption mode.
Get your key from [console.anthropic.com](https://console.anthropic.com) → API Keys.

```bash
set ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
```

### Step 4 — Start the backend

```bash
# from inside the backend/ folder
uvicorn main:app --reload --port 8000
```

You should see:
```
Loading YOLOv8 model: yolov8n.pt
Model loaded successfully!
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Step 5 — Install and start the frontend (new terminal)

```bash
cd frontend
npm install
npm run dev
```

You should see:
```
VITE v5.x.x  ready in xxx ms
➜  Local:   http://localhost:5173/
```

### Step 6 — Open the app

Go to **http://localhost:5173** in your browser.

---

## Using the App

1. **Upload** a BEFORE and AFTER image using the drag-and-drop zones
2. **Choose caption mode:**
   - `Pattern-Based` — instant, no API key required
   - `VLLM (Claude Vision)` — richer description, needs `ANTHROPIC_API_KEY`
3. Click **Analyze Changes**
4. View results:
   - Stats: objects detected before/after, added/removed/moved counts
   - Change breakdown table per object
   - Caption (with a badge showing which mode was used)
   - Side-by-side annotated image + change mask
5. **History** panel at the bottom shows all past analyses stored in SQLite

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analyze` | Upload before + after images, run analysis |
| GET | `/api/analyses` | List all past analyses |
| GET | `/api/analyses/{id}` | Get a specific analysis by ID |
| GET | `/api/health` | Health check |

**POST `/api/analyze` form fields:**
- `before` — image file (BEFORE frame)
- `after` — image file (AFTER frame)
- `caption_mode` — `"pattern"` or `"vllm"` (default: `"pattern"`)

---

## One-Click Start (Windows)

Instead of the steps above, just double-click:

```
start_backend.bat    ← starts FastAPI on port 8000
start_frontend.bat   ← installs npm deps and starts Vite on port 5173
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Object Detection | YOLOv8n (Ultralytics) |
| Change Localization | SSIM (scikit-image) |
| Frame Alignment | ORB + Homography (OpenCV) |
| VLLM Caption | Claude Haiku Vision (Anthropic) |
| Backend API | FastAPI + Uvicorn |
| Database | SQLite via SQLAlchemy |
| Frontend | React 18 + Vite |
| Image Processing | OpenCV + NumPy |

## Results
Pattern Based Caption Generation
<img width="1404" height="1029" alt="image" src="https://github.com/user-attachments/assets/4454d921-c352-4583-aefe-19071c4a1d31" />


