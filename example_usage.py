"""
Example usage script for the Change Detection System
This demonstrates how to use the system with your own images.
"""

from change_detection_system import ChangeDetectionSystem


def example_basic_usage():
    """Basic usage example."""

    print("="*70)
    print("EXAMPLE: Basic Change Detection")
    print("="*70)

    # Initialize the change detection system
    # You can use different YOLOv8 models: yolov8n.pt (fastest), yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt (most accurate)
    detector = ChangeDetectionSystem(model_name='yolov8n.pt')

    # Process two frames
    results = detector.process_frames(
        frame_a_path='Image_cc.jpg',  # Path to your first image
        frame_b_path='Image2_cc.png',  # Path to your second image
        output_a_path='output_frame_a_2.jpg',
        output_b_path='output_frame_b_2.jpg'
    )

    # Access the results
    print("\n" + "="*70)
    print("DETAILED RESULTS")
    print("="*70)

    print(f"\nNatural Language Summary:")
    print(f"  {results['description']}")

    print(f"\nAdded Objects ({len(results['changes']['added'])}):")
    for obj in results['changes']['added']:
        print(f"  - {obj['object']} at {obj['bbox']}")

    print(f"\nRemoved Objects ({len(results['changes']['removed'])}):")
    for obj in results['changes']['removed']:
        print(f"  - {obj['object']} at {obj['bbox']}")

    print(f"\nMoved Objects ({len(results['changes']['moved'])}):")
    for obj in results['changes']['moved']:
        print(f"  - {obj['object']} moved {obj['direction']} (distance: {obj['distance']:.1f} pixels)")
        print(f"    From: {obj['from_bbox']}")
        print(f"    To:   {obj['to_bbox']}")

    print(f"\nOutput images saved:")
    print(f"  - {results['output_frames'][0]}")
    print(f"  - {results['output_frames'][1]}")


def example_custom_paths():
    """Example with custom input/output paths."""

    print("\n\n" + "="*70)
    print("EXAMPLE: Custom Paths")
    print("="*70)

    detector = ChangeDetectionSystem(model_name='yolov8n.pt')

    results = detector.process_frames(
        frame_a_path='images/before.jpg',
        frame_b_path='images/after.jpg',
        output_a_path='results/before_annotated.jpg',
        output_b_path='results/after_annotated.jpg'
    )

    print(f"\nChange Summary: {results['description']}")


def example_programmatic_access():
    """Example showing step-by-step programmatic access."""

    print("\n\n" + "="*70)
    print("EXAMPLE: Step-by-Step Programmatic Access")
    print("="*70)

    import cv2

    # Initialize detector
    detector = ChangeDetectionSystem(model_name='yolov8n.pt')

    # Load images manually
    frame_a = cv2.imread('frame_a.jpg')
    frame_b = cv2.imread('frame_b.jpg')

    # Step 1: Align frames
    print("\n1. Aligning frames...")
    frame_a_aligned, frame_b_aligned = detector.preprocess_and_align(frame_a, frame_b)

    # Step 2: Compute change map
    print("2. Computing change map...")
    change_map = detector.compute_change_map(frame_a_aligned, frame_b_aligned)

    # Step 3: Detect objects
    print("3. Detecting objects...")
    detections_a = detector.detect_objects(frame_a_aligned)
    detections_b = detector.detect_objects(frame_b_aligned)
    print(f"   Found {len(detections_a)} objects in frame A")
    print(f"   Found {len(detections_b)} objects in frame B")

    # Step 4: Match objects and find changes
    print("4. Matching objects...")
    changes = detector.match_objects(detections_a, detections_b)

    # Step 5: Generate description
    print("5. Generating description...")
    description = detector.generate_description(changes)
    print(f"\n   Result: {description}")

    # Step 6: Visualize
    print("6. Creating visualizations...")
    frame_a_viz, frame_b_viz = detector.visualize_results(
        frame_a_aligned, frame_b_aligned,
        detections_a, detections_b,
        changes
    )

    # Save results
    cv2.imwrite('manual_output_a.jpg', frame_a_viz)
    cv2.imwrite('manual_output_b.jpg', frame_b_viz)
    print("Saved visualization images!")


if __name__ == "__main__":
    # Run the basic example
    # Make sure you have 'frame_a.jpg' and 'frame_b.jpg' in the same directory
    example_basic_usage()

    # Uncomment to run other examples:
    # example_custom_paths()
    # example_programmatic_access()
