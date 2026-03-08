"""
Change Detection System with Vision-Language Model (VLLM)
Enhanced version that uses BLIP-2 for natural language description generation.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim
from typing import List, Dict, Tuple
import warnings
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

warnings.filterwarnings('ignore')


class ChangeDetectionSystemVLLM:
    """
    Advanced change detection system using Vision-Language Model for
    contextual and natural description generation.
    """

    def __init__(self, yolo_model: str = 'yolov8n.pt',
                 vllm_model: str = 'Salesforce/blip2-opt-2.7b',
                 use_gpu: bool = True):
        """
        Initialize the VLLM-enhanced change detection system.

        Args:
            yolo_model: YOLOv8 model variant (yolov8n.pt, yolov8s.pt, etc.)
            vllm_model: BLIP-2 model variant
                       - 'Salesforce/blip2-opt-2.7b' (faster, 6GB VRAM)
                       - 'Salesforce/blip2-flan-t5-xl' (better quality, 12GB VRAM)
            use_gpu: Whether to use GPU for VLLM
        """
        print("="*70)
        print("INITIALIZING VLLM-ENHANCED CHANGE DETECTION SYSTEM")
        print("="*70)

        # Initialize YOLOv8 for object detection
        print(f"\n1. Loading YOLOv8 model: {yolo_model}")
        self.yolo_model = YOLO(yolo_model)
        print("   ✓ YOLOv8 loaded successfully!")

        # Initialize BLIP-2 for description generation
        print(f"\n2. Loading BLIP-2 Vision-Language Model: {vllm_model}")
        print("   (This may take a few moments...)")

        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"   Device: {self.device}")

        self.vllm_processor = Blip2Processor.from_pretrained(vllm_model)
        self.vllm_model = Blip2ForConditionalGeneration.from_pretrained(
            vllm_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.vllm_model.to(self.device)
        print("   ✓ BLIP-2 loaded successfully!")

        print("\n" + "="*70)
        print("SYSTEM READY")
        print("="*70 + "\n")

    def preprocess_and_align(self, frame_a: np.ndarray, frame_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess and align two frames to compensate for camera jitter."""
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(5000)
        keypoints_a, descriptors_a = orb.detectAndCompute(gray_a, None)
        keypoints_b, descriptors_b = orb.detectAndCompute(gray_b, None)

        if descriptors_a is not None and descriptors_b is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(descriptors_a, descriptors_b)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) > 10:
                src_pts = np.float32([keypoints_a[m.queryIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints_b[m.trainIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)

                matrix, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

                if matrix is not None:
                    height, width = frame_a.shape[:2]
                    frame_b_aligned = cv2.warpPerspective(frame_b, matrix, (width, height))
                    return frame_a, frame_b_aligned

        return frame_a, frame_b

    def compute_change_map(self, frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
        """Compute change map using SSIM."""
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

        _, diff = ssim(gray_a, gray_b, full=True)
        diff = (diff * 255).astype("uint8")

        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        return thresh

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using YOLOv8."""
        results = self.yolo_model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = results.names[class_id]

            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'class': class_name,
                'confidence': confidence,
                'center': [(x1 + x2) / 2, (y1 + y2) / 2]
            })

        return detections

    def match_objects(self, detections_a: List[Dict], detections_b: List[Dict],
                     threshold: float = 100.0) -> Dict:
        """Match objects across frames and classify changes."""
        changes = {
            'added': [],
            'removed': [],
            'moved': []
        }

        matched_b = set()

        for obj_a in detections_a:
            best_match = None
            min_distance = float('inf')

            for idx, obj_b in enumerate(detections_b):
                if idx in matched_b:
                    continue

                if obj_a['class'] == obj_b['class']:
                    distance = np.sqrt(
                        (obj_a['center'][0] - obj_b['center'][0])**2 +
                        (obj_a['center'][1] - obj_b['center'][1])**2
                    )

                    if distance < min_distance and distance < threshold:
                        min_distance = distance
                        best_match = (idx, obj_b)

            if best_match:
                matched_b.add(best_match[0])
                obj_b = best_match[1]

                if min_distance > 20:
                    dx = obj_b['center'][0] - obj_a['center'][0]
                    dy = obj_b['center'][1] - obj_a['center'][1]
                    direction = self._get_movement_direction(dx, dy)

                    changes['moved'].append({
                        'object': obj_a['class'],
                        'from_bbox': obj_a['bbox'],
                        'to_bbox': obj_b['bbox'],
                        'direction': direction,
                        'distance': min_distance
                    })
            else:
                changes['removed'].append({
                    'object': obj_a['class'],
                    'bbox': obj_a['bbox']
                })

        for idx, obj_b in enumerate(detections_b):
            if idx not in matched_b:
                changes['added'].append({
                    'object': obj_b['class'],
                    'bbox': obj_b['bbox']
                })

        return changes

    def _get_movement_direction(self, dx: float, dy: float) -> str:
        """Determine movement direction from displacement vector."""
        directions = []

        if abs(dy) > 10:
            directions.append("down" if dy > 0 else "up")

        if abs(dx) > 10:
            directions.append("right" if dx > 0 else "left")

        return "-".join(directions) if directions else "slightly"

    def generate_description_vllm(self, frame_a: np.ndarray, frame_b: np.ndarray,
                                  changes: Dict) -> str:
        """
        Generate natural language description using BLIP-2 Vision-Language Model.

        Args:
            frame_a: Reference frame (earlier)
            frame_b: Current frame (later)
            changes: Dictionary of detected changes

        Returns:
            Natural language description of changes
        """
        print("\n   Generating VLLM description...")

        # Convert OpenCV images (BGR) to PIL images (RGB)
        pil_frame_a = Image.fromarray(cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB))
        pil_frame_b = Image.fromarray(cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB))

        # Prepare structured change information
        change_summary = self._create_change_summary(changes)

        # Construct prompt for VLLM
        prompt = self._construct_prompt(changes, change_summary)

        # Process both images with BLIP-2
        # We'll analyze the second frame with context from detected changes
        inputs = self.vllm_processor(
            images=pil_frame_b,
            text=prompt,
            return_tensors="pt"
        ).to(self.device, torch.float16 if self.device == "cuda" else torch.float32)

        # Generate description
        with torch.no_grad():
            generated_ids = self.vllm_model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=5,
                temperature=0.7
            )

        description = self.vllm_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()

        # Fallback to template-based if VLLM fails
        if not description or len(description) < 10:
            print("   (VLLM output too short, using template fallback)")
            description = self._generate_template_description(changes)

        return description

    def _create_change_summary(self, changes: Dict) -> str:
        """Create a structured summary of changes for the prompt."""
        summary_parts = []

        if changes['added']:
            added_items = [c['object'] for c in changes['added']]
            summary_parts.append(f"Added: {', '.join(added_items)}")

        if changes['removed']:
            removed_items = [c['object'] for c in changes['removed']]
            summary_parts.append(f"Removed: {', '.join(removed_items)}")

        if changes['moved']:
            moved_items = [f"{c['object']} (moved {c['direction']})" for c in changes['moved']]
            summary_parts.append(f"Moved: {', '.join(moved_items)}")

        return " | ".join(summary_parts) if summary_parts else "No changes"

    def _construct_prompt(self, changes: Dict, change_summary: str) -> str:
        """
        Construct an effective prompt for BLIP-2.

        Different prompt strategies can be used based on your needs.
        """
        # Strategy 1: Guided description (Recommended for security/CCTV)
        total_changes = len(changes['added']) + len(changes['removed']) + len(changes['moved'])

        if total_changes == 0:
            return "Question: What do you see in this image? Answer:"

        # Build context-aware prompt
        prompt = "Question: This is a security camera image. "

        if changes['added']:
            objects = ', '.join([c['object'] for c in changes['added']])
            prompt += f"New objects detected: {objects}. "

        if changes['removed']:
            objects = ', '.join([c['object'] for c in changes['removed']])
            prompt += f"Objects no longer present: {objects}. "

        if changes['moved']:
            movements = [f"{c['object']} moved {c['direction']}" for c in changes['moved'][:2]]
            prompt += f"Movement detected: {', '.join(movements)}. "

        prompt += "Describe what happened in one clear sentence. Answer:"

        return prompt

    def _generate_template_description(self, changes: Dict) -> str:
        """Fallback template-based description."""
        descriptions = []

        for change in changes['added']:
            obj = change['object']
            article = 'An' if obj[0].lower() in 'aeiou' else 'A'
            descriptions.append(f"{article} {obj} was added.")

        for change in changes['removed']:
            obj = change['object']
            descriptions.append(f"{obj.capitalize()} was removed.")

        for change in changes['moved']:
            obj = change['object']
            direction = change['direction']
            descriptions.append(f"{obj.capitalize()} moved to the {direction}.")

        if not descriptions:
            return "No significant changes detected."

        return " ".join(descriptions)

    def visualize_results(self, frame_a: np.ndarray, frame_b: np.ndarray,
                         detections_a: List[Dict], detections_b: List[Dict],
                         changes: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Create visualization with bounding boxes."""
        frame_a_viz = frame_a.copy()
        frame_b_viz = frame_b.copy()

        # Draw detections on frame A
        for det in detections_a:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame_a_viz, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(frame_a_viz, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw detections on frame B with color coding
        added_bboxes = [c['bbox'] for c in changes['added']]
        moved_to_bboxes = [c['to_bbox'] for c in changes['moved']]

        for det in detections_b:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox

            if bbox in added_bboxes:
                color = (0, 255, 0)  # Green
                change_type = " [ADDED]"
            elif bbox in moved_to_bboxes:
                color = (0, 255, 255)  # Yellow
                change_type = " [MOVED]"
            else:
                color = (255, 0, 0)  # Blue
                change_type = ""

            cv2.rectangle(frame_b_viz, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']} {det['confidence']:.2f}{change_type}"
            cv2.putText(frame_b_viz, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw removed objects on frame A
        for change in changes['removed']:
            x1, y1, x2, y2 = change['bbox']
            cv2.rectangle(frame_a_viz, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{change['object']} [REMOVED]"
            cv2.putText(frame_a_viz, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame_a_viz, frame_b_viz

    def process_frames(self, frame_a_path: str, frame_b_path: str,
                      output_a_path: str = 'output_frame_a_vllm.jpg',
                      output_b_path: str = 'output_frame_b_vllm.jpg') -> Dict:
        """
        Main pipeline to process two frames with VLLM description generation.

        Args:
            frame_a_path: Path to reference frame (earlier)
            frame_b_path: Path to current frame (later)
            output_a_path: Path to save annotated frame A
            output_b_path: Path to save annotated frame B

        Returns:
            Dictionary containing changes and VLLM-generated description
        """
        print("\n" + "="*70)
        print("VLLM CHANGE DETECTION PIPELINE")
        print("="*70)

        # Load frames
        print(f"\n1. Loading frames...")
        frame_a = cv2.imread(frame_a_path)
        frame_b = cv2.imread(frame_b_path)

        if frame_a is None or frame_b is None:
            raise ValueError("Could not load one or both frames. Check file paths.")

        print(f"   Frame A: {frame_a.shape}")
        print(f"   Frame B: {frame_b.shape}")

        # Preprocess and align
        print(f"\n2. Preprocessing and aligning frames...")
        frame_a, frame_b_aligned = self.preprocess_and_align(frame_a, frame_b)
        print("   ✓ Alignment complete!")

        # Compute change map
        print(f"\n3. Computing change map using SSIM...")
        change_map = self.compute_change_map(frame_a, frame_b_aligned)
        print("   ✓ Change localization complete!")

        # Detect objects
        print(f"\n4. Detecting objects with YOLOv8...")
        detections_a = self.detect_objects(frame_a)
        detections_b = self.detect_objects(frame_b_aligned)
        print(f"   Frame A: {len(detections_a)} objects detected")
        print(f"   Frame B: {len(detections_b)} objects detected")

        # Match objects and classify changes
        print(f"\n5. Matching objects and classifying changes...")
        changes = self.match_objects(detections_a, detections_b)
        print(f"   Added: {len(changes['added'])} objects")
        print(f"   Removed: {len(changes['removed'])} objects")
        print(f"   Moved: {len(changes['moved'])} objects")

        # Generate description using VLLM
        print(f"\n6. Generating natural language description with VLLM...")
        description = self.generate_description_vllm(frame_a, frame_b_aligned, changes)
        print(f"   ✓ Description generated!")

        # Visualize results
        print(f"\n7. Creating visualizations...")
        frame_a_viz, frame_b_viz = self.visualize_results(
            frame_a, frame_b_aligned, detections_a, detections_b, changes
        )

        # Save outputs
        cv2.imwrite(output_a_path, frame_a_viz)
        cv2.imwrite(output_b_path, frame_b_viz)
        print(f"   Saved: {output_a_path}")
        print(f"   Saved: {output_b_path}")

        # Print results
        print("\n" + "="*70)
        print("VLLM RESULTS")
        print("="*70)
        print(f"\n{description}")
        print("\n" + "="*70)

        return {
            'changes': changes,
            'description': description,
            'output_frames': (output_a_path, output_b_path),
            'change_map': change_map
        }


def main():
    """Example usage of VLLM-enhanced change detection system."""

    # Initialize system with VLLM
    # Note: This requires GPU with at least 6GB VRAM for optimal performance
    detector = ChangeDetectionSystemVLLM(
        yolo_model='yolov8n.pt',
        vllm_model='Salesforce/blip2-opt-2.7b',  # Faster model
        # vllm_model='Salesforce/blip2-flan-t5-xl',  # Better quality, needs more VRAM
        use_gpu=True
    )

    # Process frames
    results = detector.process_frames(
        frame_a_path='Image_cc.jpg',
        frame_b_path='Image2_cc.png',
        output_a_path='output_frame_a_vllm.jpg',
        output_b_path='output_frame_b_vllm.jpg'
    )

    print("\n" + "="*70)
    print("DETAILED CHANGE ANALYSIS")
    print("="*70)
    print(f"\nAdded Objects: {results['changes']['added']}")
    print(f"Removed Objects: {results['changes']['removed']}")
    print(f"Moved Objects: {results['changes']['moved']}")


if __name__ == "__main__":
    main()
