"""
Wall Surface Defect Detection Module

Detects surface defects on walls (cracks, stains, paint peeling, texture anomalies,
writing/markings) using YOLOv8 fine-tuned on the Surface Inspection Defect Detection
dataset (NEU Steel, Magnetic Tile) or PatchCore for unsupervised anomaly detection.

Supported detection modes:
    - 'yolo'      : Fine-tuned YOLOv8 (recommended for labeled defect types)
    - 'patchcore'  : Unsupervised anomaly detection via anomalib (no labels needed)
    - 'texture'    : SSIM + contour-based texture change (lightweight, no training)

Surface Inspection Dataset:
    https://github.com/abin24/Surface-Inspection-defect-detection-dataset
    NEU classes: crazing, inclusion, patches, pitted_surface, rolled_in_scale, scratches
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# --------------------------------------------------------------------------- #
#  Defect class definitions                                                    #
# --------------------------------------------------------------------------- #

# NEU Steel Surface Defect dataset classes (6 classes)
NEU_CLASSES = [
    'crazing',          # Fine network of cracks  → wall: hairline cracks
    'inclusion',        # Embedded foreign material → wall: embedded debris / stains
    'patches',          # Irregular surface patches → wall: paint patches / discoloration
    'pitted_surface',   # Small pits / holes        → wall: pitting / erosion
    'rolled_in_scale',  # Rolled-in material        → wall: surface peeling / scaling
    'scratches',        # Linear surface marks      → wall: writing / graffiti / gouges
]

# Human-friendly labels for stakeholder reports
DEFECT_DISPLAY_NAMES = {
    'crazing':          'Hairline Cracks',
    'inclusion':        'Stain / Contamination',
    'patches':          'Paint Patch / Discoloration',
    'pitted_surface':   'Surface Pitting',
    'rolled_in_scale':  'Peeling / Scaling',
    'scratches':        'Writing / Scratch Marks',
    # Magnetic Tile classes (if used)
    'blowhole':         'Blowhole',
    'break':            'Surface Break',
    'crack':            'Crack',
    'fray':             'Fraying',
    'free':             'Normal',
    'uneven':           'Uneven Texture',
}

# Colour per defect class for bounding box visualisation (BGR)
DEFECT_COLORS = {
    'crazing':          (0,   165, 255),   # Orange
    'inclusion':        (255, 0,   255),   # Magenta
    'patches':          (255, 255, 0),     # Cyan
    'pitted_surface':   (0,   0,   255),   # Red
    'rolled_in_scale':  (147, 20,  255),   # Purple
    'scratches':        (0,   255, 0),     # Green
    'default':          (255, 255, 255),   # White
}


# --------------------------------------------------------------------------- #
#  Main class                                                                  #
# --------------------------------------------------------------------------- #

class WallDefectDetector:
    """
    Detects and compares surface defects on wall images.

    Two main use-cases:
        1. detect_defects(image)
               → List of defect dicts for a single image
        2. compare_wall_surfaces(image_before, image_after)
               → Dict summarising new / resolved / persisting defects
    """

    def __init__(self, mode: str = 'texture', model_path: Optional[str] = None,
                 confidence_threshold: float = 0.30):
        """
        Args:
            mode             : 'yolo' | 'patchcore' | 'texture'
            model_path       : Path to fine-tuned YOLOv8 .pt file (required for mode='yolo')
            confidence_threshold : Minimum confidence to keep a detection
        """
        self.mode = mode
        self.conf_threshold = confidence_threshold
        self.model = None

        if mode == 'yolo':
            self._load_yolo(model_path)
        elif mode == 'patchcore':
            self._load_patchcore(model_path)
        elif mode == 'texture':
            print("WallDefectDetector: texture-diff mode (no model required)")
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'yolo', 'patchcore', or 'texture'.")

    # ---------------------------------------------------------------------- #
    #  Model loading                                                           #
    # ---------------------------------------------------------------------- #

    def _load_yolo(self, model_path: Optional[str]):
        from ultralytics import YOLO
        if model_path and Path(model_path).exists():
            print(f"Loading fine-tuned YOLOv8 from: {model_path}")
            self.model = YOLO(model_path)
        else:
            print("No fine-tuned model found – loading base YOLOv8n (run fine_tune_on_dataset() first)")
            self.model = YOLO('yolov8n.pt')
        print("YOLOv8 loaded.")

    def _load_patchcore(self, model_path: Optional[str]):
        """Load anomalib PatchCore model for unsupervised anomaly detection."""
        try:
            from anomalib.models import Patchcore
            from anomalib.data.utils import read_image
            self._anomalib_read_image = read_image
            if model_path and Path(model_path).exists():
                self.model = Patchcore.load_from_checkpoint(model_path)
                print(f"PatchCore loaded from: {model_path}")
            else:
                self.model = Patchcore()
                print("PatchCore initialised (untrained – call train_patchcore() first)")
        except ImportError:
            raise ImportError(
                "anomalib is not installed. Run: pip install anomalib\n"
                "Or switch to mode='texture' for dependency-free detection."
            )

    # ---------------------------------------------------------------------- #
    #  Core detection                                                          #
    # ---------------------------------------------------------------------- #

    def detect_defects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect surface defects in a single image.

        Args:
            image : BGR image (np.ndarray from cv2.imread)

        Returns:
            List of dicts, each with keys:
                bbox        : [x1, y1, x2, y2]
                class_name  : defect class string
                display_name: human-friendly label
                confidence  : float (0-1), or 1.0 for texture mode
                center      : [cx, cy]
                area        : bounding box area in pixels
        """
        if self.mode == 'yolo':
            return self._detect_yolo(image)
        elif self.mode == 'patchcore':
            return self._detect_patchcore(image)
        else:
            return self._detect_texture(image)

    def _detect_yolo(self, image: np.ndarray) -> List[Dict]:
        results = self.model(image, verbose=False, conf=self.conf_threshold)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf  = float(box.conf[0].cpu().numpy())
            cls   = int(box.cls[0].cpu().numpy())
            name  = results.names[cls]
            detections.append(self._make_detection(
                bbox=[x1, y1, x2, y2], class_name=name, confidence=conf
            ))
        return detections

    def _detect_patchcore(self, image: np.ndarray) -> List[Dict]:
        """Run PatchCore inference and return anomaly regions as detections."""
        import torch
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        tensor = transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.no_grad():
            output = self.model(tensor)

        # anomaly_map shape: (1, H, W)
        anomaly_map = output['anomaly_map'].squeeze().numpy()
        h_orig, w_orig = image.shape[:2]
        anomaly_map = cv2.resize(anomaly_map, (w_orig, h_orig))

        # Threshold and find contours
        norm = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, thresh = cv2.threshold(norm, 180, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            score = float(anomaly_map[y:y+h, x:x+w].mean())
            detections.append(self._make_detection(
                bbox=[x, y, x+w, y+h], class_name='anomaly', confidence=min(score, 1.0)
            ))
        return detections

    def _detect_texture(self, image: np.ndarray) -> List[Dict]:
        """
        Lightweight texture-based defect detection using Canny + contour analysis.
        No model required – useful for quick demos or when no trained model exists.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Edge detection
        edges = cv2.Canny(enhanced, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Classify by shape heuristic
            aspect = w / max(h, 1)
            if aspect > 4:
                cls = 'scratches'
            elif area < 1000:
                cls = 'pitted_surface'
            else:
                cls = 'patches'
            detections.append(self._make_detection(
                bbox=[x, y, x+w, y+h], class_name=cls, confidence=0.6
            ))
        return detections

    # ---------------------------------------------------------------------- #
    #  Defect comparison (before vs after)                                    #
    # ---------------------------------------------------------------------- #

    def compare_wall_surfaces(self, image_before: np.ndarray,
                               image_after: np.ndarray,
                               iou_threshold: float = 0.3) -> Dict:
        """
        Compare defects between two images of the same wall.

        Args:
            image_before    : Reference / earlier wall image (BGR)
            image_after     : Current / later wall image (BGR)
            iou_threshold   : IoU threshold for matching same defect across frames

        Returns:
            Dict with keys:
                new_defects       : defects in 'after' not present in 'before'
                resolved_defects  : defects in 'before' no longer in 'after'
                persisting_defects: defects present in both frames
                defects_before    : all raw detections in before image
                defects_after     : all raw detections in after image
                texture_change    : dict from compute_texture_diff()
                summary           : human-readable string
        """
        defects_before = self.detect_defects(image_before)
        defects_after  = self.detect_defects(image_after)
        texture_info   = self.compute_texture_diff(image_before, image_after)

        new_defects        = []
        resolved_defects   = []
        persisting_defects = []
        matched_after      = set()

        for db in defects_before:
            best_iou  = 0.0
            best_idx  = -1
            for idx, da in enumerate(defects_after):
                if idx in matched_after:
                    continue
                iou = self._compute_iou(db['bbox'], da['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= iou_threshold:
                matched_after.add(best_idx)
                persisting_defects.append({
                    'before': db,
                    'after':  defects_after[best_idx],
                    'iou':    best_iou
                })
            else:
                resolved_defects.append(db)

        for idx, da in enumerate(defects_after):
            if idx not in matched_after:
                new_defects.append(da)

        summary = self._build_summary(new_defects, resolved_defects,
                                      persisting_defects, texture_info)

        return {
            'new_defects':        new_defects,
            'resolved_defects':   resolved_defects,
            'persisting_defects': persisting_defects,
            'defects_before':     defects_before,
            'defects_after':      defects_after,
            'texture_change':     texture_info,
            'summary':            summary,
        }

    def compute_texture_diff(self, image_before: np.ndarray,
                              image_after: np.ndarray) -> Dict:
        """
        Compute texture-level difference using SSIM and colour histogram comparison.

        Returns:
            Dict with ssim_score, color_similarity, change_percentage, change_mask
        """
        from skimage.metrics import structural_similarity as ssim

        # Resize to same shape if needed
        h, w = image_before.shape[:2]
        image_after_r = cv2.resize(image_after, (w, h))

        gray_a = cv2.cvtColor(image_before,   cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(image_after_r, cv2.COLOR_BGR2GRAY)

        score, diff = ssim(gray_a, gray_b, full=True)
        diff_uint8 = (diff * 255).astype(np.uint8)
        _, change_mask = cv2.threshold(diff_uint8, 0, 255,
                                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel)
        change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN,  kernel)

        change_pct = float(np.sum(change_mask > 0)) / change_mask.size * 100

        # Colour histogram similarity per channel
        hist_similarity = []
        for ch in range(3):
            h_a = cv2.calcHist([image_before],   [ch], None, [64], [0, 256])
            h_b = cv2.calcHist([image_after_r], [ch], None, [64], [0, 256])
            hist_similarity.append(cv2.compareHist(h_a, h_b, cv2.HISTCMP_CORREL))
        color_sim = float(np.mean(hist_similarity))

        return {
            'ssim_score':        round(score, 4),
            'color_similarity':  round(color_sim, 4),
            'change_percentage': round(change_pct, 2),
            'change_mask':       change_mask,
            'texture_changed':   score < 0.85 or color_sim < 0.90,
        }

    # ---------------------------------------------------------------------- #
    #  Visualisation                                                           #
    # ---------------------------------------------------------------------- #

    def visualize_defects(self, image: np.ndarray, defects: List[Dict],
                          title: str = '') -> np.ndarray:
        """Draw defect bounding boxes on image and return annotated copy."""
        vis = image.copy()
        for det in defects:
            x1, y1, x2, y2 = det['bbox']
            color = DEFECT_COLORS.get(det['class_name'], DEFECT_COLORS['default'])
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{det['display_name']} {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        if title:
            cv2.putText(vis, title, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return vis

    def visualize_comparison(self, image_before: np.ndarray,
                              image_after: np.ndarray,
                              comparison: Dict) -> np.ndarray:
        """
        Create a side-by-side visualisation of before/after with defect annotations.
        Green boxes  = new defects in 'after'
        Red boxes    = resolved defects (in 'before' only)
        Yellow boxes = persisting defects
        """
        h, w = image_before.shape[:2]
        image_after_r = cv2.resize(image_after, (w, h))

        vis_before = image_before.copy()
        vis_after  = image_after_r.copy()

        # Resolved defects on before image (red)
        for det in comparison['resolved_defects']:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(vis_before, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis_before, f"[RESOLVED] {det['display_name']}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Persisting defects on both images (yellow)
        for pair in comparison['persisting_defects']:
            for frame_vis, det in [(vis_before, pair['before']),
                                   (vis_after,  pair['after'])]:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame_vis, f"[EXISTING] {det['display_name']}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # New defects on after image (green)
        for det in comparison['new_defects']:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(vis_after, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_after, f"[NEW] {det['display_name']}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Labels
        cv2.putText(vis_before, "BEFORE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(vis_after,  "AFTER",  (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        return np.hstack([vis_before, vis_after])

    # ---------------------------------------------------------------------- #
    #  Fine-tuning utilities                                                   #
    # ---------------------------------------------------------------------- #

    def fine_tune_on_dataset(self, dataset_yaml: str, epochs: int = 50,
                              imgsz: int = 640, output_dir: str = 'runs/wall_defect'):
        """
        Fine-tune YOLOv8 on the Surface Inspection dataset.

        Args:
            dataset_yaml : Path to YOLO-format dataset.yaml
                           (use prepare_neu_dataset() to create this)
            epochs       : Training epochs
            imgsz        : Input image size
            output_dir   : Where to save the fine-tuned model
        """
        if self.mode != 'yolo':
            raise RuntimeError("Fine-tuning only available in mode='yolo'")
        print(f"Starting fine-tuning on {dataset_yaml} for {epochs} epochs...")
        self.model.train(data=dataset_yaml, epochs=epochs, imgsz=imgsz,
                         project=output_dir, name='wall_defect_v1', exist_ok=True)
        print(f"Training complete. Model saved to {output_dir}/wall_defect_v1/weights/best.pt")

    # ---------------------------------------------------------------------- #
    #  Helpers                                                                 #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _make_detection(bbox: List[int], class_name: str,
                        confidence: float) -> Dict:
        x1, y1, x2, y2 = bbox
        return {
            'bbox':         bbox,
            'class_name':   class_name,
            'display_name': DEFECT_DISPLAY_NAMES.get(class_name, class_name.replace('_', ' ').title()),
            'confidence':   round(confidence, 3),
            'center':       [(x1 + x2) / 2, (y1 + y2) / 2],
            'area':         (x2 - x1) * (y2 - y1),
        }

    @staticmethod
    def _compute_iou(box_a: List[int], box_b: List[int]) -> float:
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        inter_x1 = max(xa1, xb1); inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2); inter_y2 = min(ya2, yb2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union_area = ((xa2-xa1)*(ya2-ya1) + (xb2-xb1)*(yb2-yb1) - inter_area)
        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def _build_summary(new: List, resolved: List, persisting: List,
                       texture: Dict) -> str:
        lines = []
        if new:
            names = [d['display_name'] for d in new]
            lines.append(f"NEW defects detected: {', '.join(names)}.")
        if resolved:
            names = [d['display_name'] for d in resolved]
            lines.append(f"Resolved defects: {', '.join(names)}.")
        if persisting:
            lines.append(f"{len(persisting)} existing defect(s) persist unchanged.")
        if texture['texture_changed']:
            lines.append(
                f"Texture change detected: SSIM={texture['ssim_score']}, "
                f"color similarity={texture['color_similarity']}, "
                f"changed area={texture['change_percentage']:.1f}%."
            )
        if not lines:
            lines.append("No significant wall surface changes detected.")
        return ' '.join(lines)


# --------------------------------------------------------------------------- #
#  Dataset preparation utility                                                  #
# --------------------------------------------------------------------------- #

def prepare_neu_dataset(neu_root: str, output_dir: str = 'datasets/wall_defect_yolo') -> str:
    """
    Convert the NEU Surface Defect dataset to YOLO detection format.

    NEU folder structure expected:
        neu_root/
            IMAGES/       (or flat directory with .bmp/.jpg images)
            ANNOTATIONS/  (if XML annotations exist)

    If no bounding box annotations are provided (image-level labels only),
    this function creates full-image bounding boxes as a starting point for
    further annotation.

    Args:
        neu_root   : Root directory of the downloaded NEU dataset
        output_dir : Where to write the YOLO dataset

    Returns:
        Path to generated dataset.yaml
    """
    import shutil
    import random

    neu_root   = Path(neu_root)
    output_dir = Path(output_dir)

    for split in ('train', 'val'):
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Collect images grouped by class
    all_samples = []
    for idx, cls in enumerate(NEU_CLASSES):
        cls_dir = neu_root / cls
        if not cls_dir.exists():
            # Try flat directory with class prefix
            images = list(neu_root.glob(f'{cls[:2].upper()}*'))
        else:
            images = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.bmp'))
        for img_path in images:
            all_samples.append((img_path, idx))

    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.8)
    splits = {'train': all_samples[:split_idx], 'val': all_samples[split_idx:]}

    for split, samples in splits.items():
        for img_path, cls_idx in samples:
            # Copy image
            dst_img = output_dir / 'images' / split / img_path.name
            shutil.copy(img_path, dst_img)

            # Write full-image bounding box label (normalised YOLO format)
            label_path = output_dir / 'labels' / split / (img_path.stem + '.txt')
            with open(label_path, 'w') as f:
                f.write(f"{cls_idx} 0.5 0.5 1.0 1.0\n")  # cx cy w h

    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"path: {output_dir.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(NEU_CLASSES)}\n")
        f.write(f"names: {NEU_CLASSES}\n")

    print(f"Dataset prepared at {output_dir}. YAML: {yaml_path}")
    print(f"Total samples: {len(all_samples)} ({split_idx} train / {len(all_samples)-split_idx} val)")
    return str(yaml_path)


# --------------------------------------------------------------------------- #
#  Quick standalone demo                                                        #
# --------------------------------------------------------------------------- #

def main():
    """Quick demo: run texture-mode detection on two test images."""
    import sys

    img_a_path = sys.argv[1] if len(sys.argv) > 1 else 'Image_cc.jpg'
    img_b_path = sys.argv[2] if len(sys.argv) > 2 else 'Image2_cc.png'

    img_a = cv2.imread(img_a_path)
    img_b = cv2.imread(img_b_path)

    if img_a is None or img_b is None:
        print("Could not load images. Provide paths as arguments: python wall_defect_detector.py <before> <after>")
        return

    detector = WallDefectDetector(mode='texture')
    comparison = detector.compare_wall_surfaces(img_a, img_b)

    print("\n" + "="*60)
    print("WALL SURFACE ANALYSIS")
    print("="*60)
    print(comparison['summary'])
    print(f"\nNew defects     : {len(comparison['new_defects'])}")
    print(f"Resolved defects: {len(comparison['resolved_defects'])}")
    print(f"Persisting      : {len(comparison['persisting_defects'])}")

    vis = detector.visualize_comparison(img_a, img_b, comparison)
    cv2.imwrite('wall_surface_comparison.jpg', vis)
    print("\nVisualisation saved to wall_surface_comparison.jpg")


if __name__ == '__main__':
    main()
