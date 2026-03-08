"""
Integrated Change Detection + Wall Surface Defect System

Combines:
    1. Object-level change detection  (ChangeDetectionSystem)
    2. Wall surface defect detection  (WallDefectDetector)

into a single unified pipeline that produces:
    - Object-level changes (added / removed / moved)
    - Wall surface defect changes (new / resolved / persisting defects)
    - Texture-level analysis (SSIM, colour shift, change %)
    - A combined natural-language summary
    - Annotated output images

Usage:
    from integrated_system import IntegratedSystem
    system = IntegratedSystem()
    results = system.process_frames('before.jpg', 'after.jpg')
    print(results['combined_summary'])
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from change_detection_system import ChangeDetectionSystem
from wall_defect_detector import WallDefectDetector


class IntegratedSystem:
    """
    Unified pipeline for frame-level change detection AND wall surface analysis.

    Args:
        object_model    : YOLOv8 model for object detection (e.g. 'yolov8n.pt')
        defect_mode     : Wall defect detector mode – 'texture' | 'yolo' | 'patchcore'
        defect_model    : Path to fine-tuned defect detection model (optional)
        conf_threshold  : Confidence threshold for defect detection
    """

    def __init__(self,
                 object_model: str = 'yolov8n.pt',
                 defect_mode: str = 'texture',
                 defect_model: Optional[str] = None,
                 conf_threshold: float = 0.30):

        print("="*60)
        print("Initialising Integrated Change Detection System")
        print("="*60)

        print("\n[1/2] Loading object-level change detector...")
        self.change_detector = ChangeDetectionSystem(model_name=object_model)

        print("\n[2/2] Loading wall surface defect detector...")
        self.defect_detector = WallDefectDetector(
            mode=defect_mode,
            model_path=defect_model,
            confidence_threshold=conf_threshold
        )
        print("\nSystem ready.\n")

    # ---------------------------------------------------------------------- #
    #  Main pipeline                                                           #
    # ---------------------------------------------------------------------- #

    def process_frames(self, frame_a_path: str, frame_b_path: str,
                       output_dir: str = 'outputs') -> Dict:
        """
        Run the full integrated pipeline on two images.

        Args:
            frame_a_path : Path to the reference / 'before' frame
            frame_b_path : Path to the current / 'after' frame
            output_dir   : Directory to save output images

        Returns:
            Dict with keys:
                object_changes    : from ChangeDetectionSystem
                surface_changes   : from WallDefectDetector
                combined_summary  : unified natural-language description
                output_paths      : dict of saved image paths
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print("\n" + "="*60)
        print("INTEGRATED CHANGE DETECTION PIPELINE")
        print("="*60)

        # ------------------------------------------------------------------ #
        # Step 1 – Load images                                                #
        # ------------------------------------------------------------------ #
        print("\n[Step 1] Loading images...")
        frame_a = cv2.imread(frame_a_path)
        frame_b = cv2.imread(frame_b_path)
        if frame_a is None or frame_b is None:
            raise ValueError("Could not load one or both frames. Check file paths.")
        print(f"  Before : {frame_a_path}  {frame_a.shape}")
        print(f"  After  : {frame_b_path}  {frame_b.shape}")

        # ------------------------------------------------------------------ #
        # Step 2 – Frame alignment                                            #
        # ------------------------------------------------------------------ #
        print("\n[Step 2] Aligning frames (ORB + Homography)...")
        frame_a_aligned, frame_b_aligned = self.change_detector.preprocess_and_align(
            frame_a, frame_b
        )
        print("  Alignment complete.")

        # ------------------------------------------------------------------ #
        # Step 3 – Object-level change detection                              #
        # ------------------------------------------------------------------ #
        print("\n[Step 3] Detecting object-level changes (YOLOv8 + SSIM)...")
        change_map   = self.change_detector.compute_change_map(frame_a_aligned, frame_b_aligned)
        detections_a = self.change_detector.detect_objects(frame_a_aligned)
        detections_b = self.change_detector.detect_objects(frame_b_aligned)
        object_changes = self.change_detector.match_objects(detections_a, detections_b)
        object_description = self.change_detector.generate_description(object_changes)

        print(f"  Objects in before : {len(detections_a)}")
        print(f"  Objects in after  : {len(detections_b)}")
        print(f"  Added    : {len(object_changes['added'])}")
        print(f"  Removed  : {len(object_changes['removed'])}")
        print(f"  Moved    : {len(object_changes['moved'])}")

        # ------------------------------------------------------------------ #
        # Step 4 – Wall surface defect detection                              #
        # ------------------------------------------------------------------ #
        print("\n[Step 4] Analysing wall surface defects...")
        surface_changes = self.defect_detector.compare_wall_surfaces(
            frame_a_aligned, frame_b_aligned
        )

        print(f"  New defects       : {len(surface_changes['new_defects'])}")
        print(f"  Resolved defects  : {len(surface_changes['resolved_defects'])}")
        print(f"  Persisting defects: {len(surface_changes['persisting_defects'])}")
        texture = surface_changes['texture_change']
        print(f"  SSIM score        : {texture['ssim_score']}")
        print(f"  Colour similarity : {texture['color_similarity']}")
        print(f"  Changed area      : {texture['change_percentage']}%")

        # ------------------------------------------------------------------ #
        # Step 5 – Build combined summary                                     #
        # ------------------------------------------------------------------ #
        combined_summary = self._build_combined_summary(
            object_description, surface_changes, texture
        )

        # ------------------------------------------------------------------ #
        # Step 6 – Visualise and save outputs                                 #
        # ------------------------------------------------------------------ #
        print("\n[Step 5] Generating visualisations...")
        output_paths = self._save_outputs(
            frame_a_aligned, frame_b_aligned,
            detections_a, detections_b, object_changes,
            surface_changes, change_map,
            output_dir
        )

        # ------------------------------------------------------------------ #
        # Print final results                                                  #
        # ------------------------------------------------------------------ #
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(combined_summary)
        print("\nOutput files:")
        for key, path in output_paths.items():
            print(f"  {key:30s}: {path}")
        print("="*60)

        return {
            'object_changes':   object_changes,
            'surface_changes':  surface_changes,
            'combined_summary': combined_summary,
            'output_paths':     output_paths,
        }

    # ---------------------------------------------------------------------- #
    #  Summary builder                                                         #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _build_combined_summary(object_desc: str, surface: Dict,
                                 texture: Dict) -> str:
        sections = []

        # Object section
        if object_desc != "No significant changes detected.":
            sections.append(f"Object-Level Changes:\n  {object_desc}")

        # Surface defect section
        defect_lines = []
        if surface['new_defects']:
            names = [d['display_name'] for d in surface['new_defects']]
            defect_lines.append(f"  NEW surface defects: {', '.join(names)}")
        if surface['resolved_defects']:
            names = [d['display_name'] for d in surface['resolved_defects']]
            defect_lines.append(f"  Resolved defects: {', '.join(names)}")
        if surface['persisting_defects']:
            defect_lines.append(
                f"  {len(surface['persisting_defects'])} existing defect(s) unchanged"
            )
        if defect_lines:
            sections.append("Wall Surface Changes:\n" + "\n".join(defect_lines))

        # Texture section
        tex_lines = []
        if texture['texture_changed']:
            tex_lines.append(
                f"  Texture/paint change detected  "
                f"(SSIM: {texture['ssim_score']}, "
                f"colour similarity: {texture['color_similarity']}, "
                f"affected area: {texture['change_percentage']:.1f}%)"
            )
        else:
            tex_lines.append(
                f"  Wall texture/paint appears stable  "
                f"(SSIM: {texture['ssim_score']}, "
                f"colour similarity: {texture['color_similarity']})"
            )
        sections.append("Texture Analysis:\n" + "\n".join(tex_lines))

        if not sections:
            return "No significant changes detected between the two frames."

        return "\n\n".join(sections)

    # ---------------------------------------------------------------------- #
    #  Output saving                                                           #
    # ---------------------------------------------------------------------- #

    def _save_outputs(self, frame_a, frame_b,
                      detections_a, detections_b, object_changes,
                      surface_changes, change_map, output_dir) -> Dict:
        """Save all annotated output images and return their paths."""
        out = Path(output_dir)

        # 1. Object change visualisation (before / after with YOLO boxes)
        obj_vis_a, obj_vis_b = self.change_detector.visualize_results(
            frame_a, frame_b, detections_a, detections_b, object_changes
        )
        path_obj_a = str(out / 'object_change_before.jpg')
        path_obj_b = str(out / 'object_change_after.jpg')
        cv2.imwrite(path_obj_a, obj_vis_a)
        cv2.imwrite(path_obj_b, obj_vis_b)

        # 2. Wall defect comparison (side-by-side)
        defect_vis = self.defect_detector.visualize_comparison(
            frame_a, frame_b, surface_changes
        )
        path_defect = str(out / 'wall_defect_comparison.jpg')
        cv2.imwrite(path_defect, defect_vis)

        # 3. Texture change mask
        path_mask = str(out / 'texture_change_mask.jpg')
        cv2.imwrite(path_mask, change_map)

        # 4. Combined 2x2 summary panel
        summary_panel = self._build_summary_panel(
            obj_vis_a, obj_vis_b, defect_vis, change_map
        )
        path_panel = str(out / 'integrated_summary.jpg')
        cv2.imwrite(path_panel, summary_panel)

        return {
            'object_change_before':    path_obj_a,
            'object_change_after':     path_obj_b,
            'wall_defect_comparison':  path_defect,
            'texture_change_mask':     path_mask,
            'integrated_summary':      path_panel,
        }

    @staticmethod
    def _build_summary_panel(obj_a: np.ndarray, obj_b: np.ndarray,
                              defect_vis: np.ndarray,
                              change_mask: np.ndarray) -> np.ndarray:
        """Build a 2×2 grid summary panel."""
        TARGET_W = 640
        TARGET_H = 360

        def resize(img, w=TARGET_W, h=TARGET_H):
            return cv2.resize(img, (w, h))

        # Defect comparison is already side-by-side, split it back
        dw = defect_vis.shape[1] // 2
        def_before = resize(defect_vis[:, :dw])
        def_after  = resize(defect_vis[:, dw:])

        # Change mask → 3-channel colourmap
        mask_color = cv2.applyColorMap(
            cv2.resize(change_mask, (TARGET_W, TARGET_H)), cv2.COLORMAP_HOT
        )

        top_row    = np.hstack([resize(obj_a), resize(obj_b)])
        bottom_row = np.hstack([def_before, def_after])

        panel = np.vstack([top_row, bottom_row])

        # Add row labels
        cv2.putText(panel, "Object Change Detection",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(panel, "Wall Surface Defect Analysis",
                    (10, TARGET_H + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        return panel


# --------------------------------------------------------------------------- #
#  Standalone entry point                                                       #
# --------------------------------------------------------------------------- #

def main():
    import sys

    before = sys.argv[1] if len(sys.argv) > 1 else 'Image_cc.jpg'
    after  = sys.argv[2] if len(sys.argv) > 2 else 'Image2_cc.png'

    system = IntegratedSystem(defect_mode='texture')
    results = system.process_frames(before, after, output_dir='outputs')

    print("\nDetailed object changes:")
    print(f"  Added   : {results['object_changes']['added']}")
    print(f"  Removed : {results['object_changes']['removed']}")
    print(f"  Moved   : {results['object_changes']['moved']}")


if __name__ == '__main__':
    main()
