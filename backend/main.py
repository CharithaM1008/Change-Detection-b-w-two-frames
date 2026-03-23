"""
Change Detection System - FastAPI Backend
Uses ChangeDetectionSystem (YOLO) for object-level change detection.
Supports pattern-based and VLLM (Claude Vision) caption generation.
"""

import sys
import json
import uuid
import base64
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.append(str(Path(__file__).parent.parent))
from change_detection_system import ChangeDetectionSystem

from database import SessionLocal, engine
import models

models.Base.metadata.create_all(bind=engine)

UPLOAD_DIR = Path(__file__).parent / "uploads"
RESULTS_DIR = Path(__file__).parent / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Change Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

# Load YOLO model once at startup
detector = ChangeDetectionSystem(model_name="yolov8n.pt")


def _img_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


def _generate_vllm_caption(
    img_before: np.ndarray,
    img_after: np.ndarray,
    changes: dict,
) -> str:
    """Send both frames to Claude Vision and get a detailed change description."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "VLLM unavailable: ANTHROPIC_API_KEY not set."
    try:
        import anthropic

        pattern_hint = (
            f"Automated detection found: {len(changes['added'])} object(s) added, "
            f"{len(changes['removed'])} removed, {len(changes['moved'])} moved."
        )

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": _img_to_b64(img_before),
                            },
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": _img_to_b64(img_after),
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "These are two frames from a security / surveillance camera. "
                                "The first image is BEFORE and the second is AFTER. "
                                f"Context: {pattern_hint} "
                                "Describe in 2-3 sentences exactly what changed between the two frames. "
                                "Be specific: mention the objects, their positions, and what happened."
                            ),
                        },
                    ],
                }
            ],
        )
        return message.content[0].text.strip()
    except Exception as exc:
        return f"VLLM error: {exc}"


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
    caption_mode: str = Form("pattern"),
):
    uid = str(uuid.uuid4())[:8]
    before_ext = Path(before.filename).suffix or ".jpg"
    after_ext = Path(after.filename).suffix or ".jpg"

    before_fn = f"before_{uid}{before_ext}"
    after_fn = f"after_{uid}{after_ext}"
    before_path = UPLOAD_DIR / before_fn
    after_path = UPLOAD_DIR / after_fn

    with open(before_path, "wb") as f:
        f.write(await before.read())
    with open(after_path, "wb") as f:
        f.write(await after.read())

    img_before = cv2.imread(str(before_path))
    img_after = cv2.imread(str(after_path))

    if img_before is None or img_after is None:
        raise HTTPException(400, "Could not decode one or both images.")

    # Pipeline
    img_a, img_b = detector.preprocess_and_align(img_before, img_after)
    change_map = detector.compute_change_map(img_a, img_b)
    dets_a = detector.detect_objects(img_a)
    dets_b = detector.detect_objects(img_b)
    changes = detector.match_objects(dets_a, dets_b)

    # Caption
    if caption_mode == "vllm":
        caption = _generate_vllm_caption(img_a, img_b, changes)
    else:
        caption = detector.generate_description(changes)

    # Visualize — side-by-side
    frame_a_viz, frame_b_viz = detector.visualize_results(img_a, img_b, dets_a, dets_b, changes)
    vis = np.hstack([frame_a_viz, frame_b_viz])

    result_fn = f"result_{uid}.jpg"
    mask_fn = f"mask_{uid}.jpg"
    cv2.imwrite(str(RESULTS_DIR / result_fn), vis)
    cv2.imwrite(str(RESULTS_DIR / mask_fn), change_map)

    result_data = {
        "changes": changes,
        "caption": caption,
        "caption_mode": caption_mode,
        "stats": {
            "objects_before": len(dets_a),
            "objects_after": len(dets_b),
            "added": len(changes["added"]),
            "removed": len(changes["removed"]),
            "moved": len(changes["moved"]),
        },
    }

    db = SessionLocal()
    record = models.Analysis(
        before_filename=before_fn,
        after_filename=after_fn,
        result_filename=result_fn,
        mask_filename=mask_fn,
        added_count=len(changes["added"]),
        removed_count=len(changes["removed"]),
        moved_count=len(changes["moved"]),
        caption_mode=caption_mode,
        caption=caption,
        full_result=json.dumps(result_data),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    db.close()

    return {
        "id": record.id,
        "created_at": record.created_at.isoformat(),
        "before_url": f"/uploads/{before_fn}",
        "after_url": f"/uploads/{after_fn}",
        "result_url": f"/results/{result_fn}",
        "mask_url": f"/results/{mask_fn}",
        **result_data,
    }


@app.get("/api/analyses")
def list_analyses():
    db = SessionLocal()
    rows = (
        db.query(models.Analysis)
        .order_by(models.Analysis.created_at.desc())
        .limit(30)
        .all()
    )
    db.close()
    return [
        {
            "id": r.id,
            "created_at": r.created_at.isoformat(),
            "added_count": r.added_count,
            "removed_count": r.removed_count,
            "moved_count": r.moved_count,
            "caption_mode": r.caption_mode,
            "caption": r.caption,
            "result_url": f"/results/{r.result_filename}",
        }
        for r in rows
    ]


@app.get("/api/analyses/{analysis_id}")
def get_analysis(analysis_id: int):
    db = SessionLocal()
    r = db.query(models.Analysis).filter(models.Analysis.id == analysis_id).first()
    db.close()
    if not r:
        raise HTTPException(404, "Analysis not found.")
    data = json.loads(r.full_result)
    return {
        "id": r.id,
        "created_at": r.created_at.isoformat(),
        "before_url": f"/uploads/{r.before_filename}",
        "after_url": f"/uploads/{r.after_filename}",
        "result_url": f"/results/{r.result_filename}",
        "mask_url": f"/results/{r.mask_filename}",
        **data,
    }
