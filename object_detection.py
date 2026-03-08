"""
Simple Object Detection Script
Detects objects in an image and draws rectangular bounding boxes around them.
"""

import cv2
from ultralytics import YOLO


class ObjectDetector:
    """Simple object detector using YOLOv8."""

    def __init__(self, model_name: str = 'yolov8n.pt'):
        """
        Initialize the object detector.

        Args:
            model_name: YOLOv8 model to use
                       - yolov8n.pt (nano - fastest, smallest)
                       - yolov8s.pt (small)
                       - yolov8m.pt (medium)
                       - yolov8l.pt (large)
                       - yolov8x.pt (extra large - most accurate)
        """
        print(f"Loading YOLOv8 model: {model_name}")
        self.model = YOLO(model_name)
        print("Model loaded successfully!")

    def detect_objects(self, image_path: str, output_path: str = None,
                      confidence_threshold: float = 0.25) -> dict:
        """
        Detect objects in an image and draw bounding boxes.

        Args:
            image_path: Path to input image
            output_path: Path to save output image (if None, auto-generated)
            confidence_threshold: Minimum confidence for detection (0.0 to 1.0)

        Returns:
            Dictionary containing detections and output path
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        print(f"\nProcessing image: {image_path}")
        print(f"Image size: {image.shape}")

        # Run detection
        results = self.model(image, conf=confidence_threshold, verbose=False)[0]

        # Prepare output
        detections = []
        annotated_image = image.copy()

        # Draw bounding boxes
        for box in results.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            # Get confidence and class
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = results.names[class_id]

            # Store detection info
            detections.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })

            # Draw rectangle
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label with background
            label = f"{class_name} {confidence:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            # Draw label background
            cv2.rectangle(
                annotated_image,
                (x1, y1 - label_height - baseline - 5),
                (x1 + label_width, y1),
                (0, 255, 0),
                -1
            )

            # Draw label text
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )

        # Generate output path if not provided
        if output_path is None:
            output_path = image_path.rsplit('.', 1)[0] + '_detected.jpg'

        # Save annotated image
        cv2.imwrite(output_path, annotated_image)

        # Print summary
        print(f"\nDetection Summary:")
        print(f"  Total objects detected: {len(detections)}")
        print(f"  Output saved to: {output_path}")

        print(f"\nDetected objects:")
        for i, det in enumerate(detections, 1):
            print(f"  {i}. {det['class']} (confidence: {det['confidence']:.2f}) at {det['bbox']}")

        return {
            'detections': detections,
            'output_path': output_path,
            'total_objects': len(detections)
        }


def main():
    """Example usage of object detector."""

    # Initialize detector
    detector = ObjectDetector(model_name='yolov8n.pt')

    # Detect objects in a single image
    result = detector.detect_objects(
        image_path='Image_cc.jpg',
        output_path='Image_detected.jpg',
        confidence_threshold=0.25  # Adjust this to filter low-confidence detections
    )

    # You can also detect in multiple images
    print("\n" + "="*70)
    print("Processing second image...")
    print("="*70)

    result2 = detector.detect_objects(
        image_path='Image2.png',
        output_path='Image2_detected.jpg',
        confidence_threshold=0.25
    )


if __name__ == "__main__":
    main()
