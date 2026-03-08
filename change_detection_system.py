"""
Change Detection System between Two Frames
An AI-powered system that detects, classifies, and explains object-level changes
between two image frames for CCTV and security surveillance.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ChangeDetectionSystem:
    """Main class for detecting and classifying changes between two image frames."""

    def __init__(self, model_name: str = 'yolov8n.pt'):
        """
        Initialize the change detection system.

        Args:
            model_name: YOLOv8 model variant to use (yolov8n.pt, yolov8s.pt, etc.)
        """ 
        print(f"Loading YOLOv8 model: {model_name}")
        self.model = YOLO(model_name)
        print("Model loaded successfully!")

    def preprocess_and_align(self, frame_a: np.ndarray, frame_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess and align two frames to compensate for camera jitter.

        Args:
            frame_a: Reference image (earlier frame)
            frame_b: Current image (later frame)

        Returns:
            Tuple of aligned frames (frame_a, frame_b_aligned)
        """
        # Convert to grayscale for alignment
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

        # Detect ORB keypoints and descriptors
        orb = cv2.ORB_create(5000)
        keypoints_a, descriptors_a = orb.detectAndCompute(gray_a, None)
        keypoints_b, descriptors_b = orb.detectAndCompute(gray_b, None)

        # Match features
        if descriptors_a is not None and descriptors_b is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(descriptors_a, descriptors_b)
            matches = sorted(matches, key=lambda x: x.distance)

            # Use top matches to estimate transformation
            if len(matches) > 10:
                src_pts = np.float32([keypoints_a[m.queryIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints_b[m.trainIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)

                # Find homography matrix
                matrix, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

                if matrix is not None:
                    # Warp frame_b to align with frame_a
                    height, width = frame_a.shape[:2]
                    frame_b_aligned = cv2.warpPerspective(frame_b, matrix, (width, height))
                    return frame_a, frame_b_aligned

        # If alignment fails, return original frames
        return frame_a, frame_b

    def compute_change_map(self, frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
        """
        Compute change map using SSIM to detect meaningful structural changes.

        Args:
            frame_a: Reference image
            frame_b: Current image

        Returns:
            Binary change mask
        """
        # Convert to grayscale
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

        # Compute SSIM
        score, diff = ssim(gray_a, gray_b, full=True)
        diff = (diff * 255).astype("uint8")

        # Threshold the difference image
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        return thresh

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects and people in a frame using YOLOv8.

        Args:
            frame: Input image

        Returns:
            List of detected objects with bounding boxes and class information
        """
        results = self.model(frame, verbose=False)[0]
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
        """
        Match objects across frames and classify changes.

        Args:
            detections_a: Detections from frame A
            detections_b: Detections from frame B
            threshold: Maximum distance for matching objects

        Returns:
            Dictionary containing added, removed, and moved objects
        """
        changes = {
            'added': [],
            'removed': [],
            'moved': []
        }

        matched_b = set()

        # Find moved objects and objects in frame A
        for obj_a in detections_a:
            best_match = None
            min_distance = float('inf')

            for idx, obj_b in enumerate(detections_b):
                if idx in matched_b:
                    continue

                # Match by class and proximity
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

                # Check if object moved significantly
                if min_distance > 20:  # Movement threshold
                    dx = obj_b['center'][0] - obj_a['center'][0]
                    dy = obj_b['center'][1] - obj_a['center'][1]

                    # Determine primary direction
                    direction = self._get_movement_direction(dx, dy)

                    changes['moved'].append({
                        'object': obj_a['class'],
                        'from_bbox': obj_a['bbox'],
                        'to_bbox': obj_b['bbox'],
                        'direction': direction,
                        'distance': min_distance
                    })
            else:
                # Object in A but not in B = removed
                changes['removed'].append({
                    'object': obj_a['class'],
                    'bbox': obj_a['bbox']
                })

        # Find added objects (in B but not matched with A)
        for idx, obj_b in enumerate(detections_b):
            if idx not in matched_b:
                changes['added'].append({
                    'object': obj_b['class'],
                    'bbox': obj_b['bbox']
                })

        return changes

    def _get_movement_direction(self, dx: float, dy: float) -> str:
        """
        Determine movement direction from displacement vector.

        Args:
            dx: Horizontal displacement
            dy: Vertical displacement

        Returns:
            Direction string (e.g., "right", "up-left")
        """
        directions = []

        # Vertical direction
        if abs(dy) > 10:
            directions.append("down" if dy > 0 else "up")

        # Horizontal direction
        if abs(dx) > 10:
            directions.append("right" if dx > 0 else "left")

        return "-".join(directions) if directions else "slightly"

    def generate_description(self, changes: Dict) -> str:
        """
        Generate natural language description of changes.

        Args:
            changes: Dictionary of detected changes

        Returns:
            Human-readable description string
        """
        descriptions = []

        # Added objects
        for change in changes['added']:
            obj = change['object']
            article = 'An' if obj[0].lower() in 'aeiou' else 'A'
            descriptions.append(f"{article} {obj} was added.")

        # Removed objects
        for change in changes['removed']:
            obj = change['object']
            descriptions.append(f"{obj.capitalize()} was removed.")

        # Moved objects
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
        """
        Create visualization with bounding boxes for detected objects.

        Args:
            frame_a: Reference frame
            frame_b: Current frame
            detections_a: Detections in frame A
            detections_b: Detections in frame B
            changes: Detected changes

        Returns:
            Tuple of annotated frames (frame_a_viz, frame_b_viz)
        """
        frame_a_viz = frame_a.copy()
        frame_b_viz = frame_b.copy()

        # Draw all detections on frame A (in blue)
        for det in detections_a:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame_a_viz, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(frame_a_viz, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw all detections on frame B with color coding
        removed_bboxes = [c['bbox'] for c in changes['removed']]
        added_bboxes = [c['bbox'] for c in changes['added']]
        moved_to_bboxes = [c['to_bbox'] for c in changes['moved']]

        for det in detections_b:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox

            # Color code: Green for added, Yellow for moved, Blue for unchanged
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

        # Draw removed objects on frame A (in red)
        for change in changes['removed']:
            x1, y1, x2, y2 = change['bbox']
            cv2.rectangle(frame_a_viz, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{change['object']} [REMOVED]"
            cv2.putText(frame_a_viz, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame_a_viz, frame_b_viz

    def process_frames(self, frame_a_path: str, frame_b_path: str,
                      output_a_path: str = 'output_frame_a.jpg',
                      output_b_path: str = 'output_frame_b.jpg') -> Dict:
        """
        Main pipeline to process two frames and detect changes.

        Args:
            frame_a_path: Path to reference frame (earlier)
            frame_b_path: Path to current frame (later)
            output_a_path: Path to save annotated frame A
            output_b_path: Path to save annotated frame B

        Returns:
            Dictionary containing changes and description
        """
        print("\n" + "="*60)
        print("CHANGE DETECTION SYSTEM")
        print("="*60)

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
        print("   Alignment complete!")

        # Compute change map
        print(f"\n3. Computing change map using SSIM...")
        change_map = self.compute_change_map(frame_a, frame_b_aligned)
        print("   Change localization complete!")

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

        # Generate description
        print(f"\n6. Generating natural language description...")
        description = self.generate_description(changes)

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
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"\n{description}")
        print("\n" + "="*60)

        return {
            'changes': changes,
            'description': description,
            'output_frames': (output_a_path, output_b_path)
        }


def main():
    """Example usage of the change detection system."""

    # Initialize system
    detector = ChangeDetectionSystem(model_name='yolov8n.pt')

    # Process frames
    results = detector.process_frames(
        frame_a_path='frame_a.jpg',
        frame_b_path='frame_b.jpg',
        output_a_path='output_frame_a.jpg',
        output_b_path='output_frame_b.jpg'
    )

    print("\nDetailed Changes:")
    print(f"Added: {results['changes']['added']}")
    print(f"Removed: {results['changes']['removed']}")
    print(f"Moved: {results['changes']['moved']}")


if __name__ == "__main__":
    main()
