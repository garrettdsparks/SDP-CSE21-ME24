"""Fire Detection Module for Industrial Boiler Room Monitoring.

This module provides detection capabilities using YOLO (You Only Look Once).
It processes static images or video streams frame-by-frame, tracking detected
fire/smoke hazards with confidence filtering and temporal persistence.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Generator, Tuple
import cv2
import numpy as np


class FireDetector:
    """Detects fire and hazards in images and video frames using a YOLO model.

    Attributes:
        model_path (str): Path to trained YOLO weights (.pt file).
        conf_threshold (float): Minimum confidence required to accept a detection (0.0 to 1.0).
        model: Loaded YOLO model instance from Ultralytics.
    """

    def __init__(self, model_path: str = "yolo11n.pt", conf_threshold: float = 0.25):
        """Initializes the FireDetector with a model file and confidence threshold."""
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads the YOLO model into memory using Ultralytics."""
        try:
            from ultralytics import YOLO
            print(f"[INFO] Loading YOLO model weights from '{self.model_path}'...")
            self.model = YOLO(self.model_path)
            print("[INFO] Model loaded successfully.")
        except ImportError:
            print("[ERROR] 'ultralytics' package is not installed. Please run: pip install ultralytics", file=sys.stderr)
            raise

    def detect_image(self, image_input: np.ndarray | str) -> Tuple[np.ndarray, list]:
        """Runs detection on a single image.

        Args:
            image_input: File path (str) or loaded BGR image (np.ndarray).

        Returns:
            Tuple of:
                - annotated_image (np.ndarray): Image with bounding boxes and labels drawn.
                - detections (list): List of dicts containing box coordinates, class_name, and confidence.
        """
        results = self.model.predict(
            source=image_input,
            conf=self.conf_threshold,
            verbose=False
        )

        detections = []
        first_result = results[0]

        # Extract detected bounding boxes
        for box in first_result.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = self.model.names.get(cls_id, str(cls_id))
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

            detections.append({
                "class_name": cls_name,
                "confidence": conf,
                "box": xyxy  # [xmin, ymin, xmax, ymax]
            })

        annotated_image = first_result.plot()
        return annotated_image, detections

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        persistence_frames: int = 3,
        display: bool = False
    ) -> Generator[Tuple[int, int, list], None, None]:
        """Processes a video file frame-by-frame with a temporal persistence filter.

        Why Persistence Matters in Industrial Safety:
            A single-frame flicker or reflection might trigger a false alarm.
            A persistence filter requires an alert to be detected across multiple
            consecutive frames (e.g., >= 3 frames) before triggering an official alarm.

        Args:
            video_path: Path to the input video file (e.g., .mp4, .avi).
            output_path: Optional path to save the annotated output video.
            persistence_frames: Number of consecutive frames required to confirm a hazard.
            display: Whether to open an OpenCV window displaying the live stream.

        Yields:
            Tuple of (frame_number, consecutive_hazard_count, detections)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        consecutive_hazard_count = 0
        frame_idx = 0

        print(f"[INFO] Processing video: {video_path} ({total_frames} frames @ {fps:.1f} FPS)")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                annotated_frame, detections = self.detect_image(frame)

                # Check if any hazard/fire was detected in this frame
                has_hazard = len(detections) > 0

                if has_hazard:
                    consecutive_hazard_count += 1
                else:
                    consecutive_hazard_count = max(0, consecutive_hazard_count - 1)

                # If hazard persists beyond threshold, flag confirmed alarm
                is_confirmed_alarm = consecutive_hazard_count >= persistence_frames

                # Visual banner for confirmed alert
                if is_confirmed_alarm:
                    cv2.putText(
                        annotated_frame,
                        "CRITICAL ALERT: HAZARD CONFIRMED!",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3,
                        cv2.LINE_AA
                    )

                if writer:
                    writer.write(annotated_frame)

                if display:
                    cv2.imshow("Industrial Boiler Room Hazard Monitor", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[INFO] Video playback stopped by user.")
                        break

                yield (frame_idx, consecutive_hazard_count, detections)

        finally:
            cap.release()
            if writer:
                writer.release()
                print(f"[INFO] Annotated video saved to: {output_path}")
            if display:
                cv2.destroyAllWindows()


def main():
    """Command-line interface for the fire detector."""
    parser = argparse.ArgumentParser(description="Industrial Boiler Room Hazard & Fire Detector")
    parser.add_argument("--source", type=str, required=True, help="Path to input image or video file")
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="Path to YOLO weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--output", type=str, default=None, help="Path to save output result")
    parser.add_argument("--display", action="store_true", help="Display video stream window")
    parser.add_argument("--persistence", type=int, default=3, help="Consecutive frames for confirmed alarm")

    args = parser.parse_args()
    source_path = Path(args.source)

    if not source_path.exists():
        print(f"[ERROR] Source file does not exist: {source_path}", file=sys.stderr)
        sys.exit(1)

    detector = FireDetector(model_path=args.weights, conf_threshold=args.conf)

    # Distinguish between video and image files by extension
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    if source_path.suffix.lower() in video_extensions:
        print("[INFO] Running in Video Stream mode...")
        for frame_num, hazard_count, detections in detector.process_video(
            video_path=str(source_path),
            output_path=args.output,
            persistence_frames=args.persistence,
            display=args.display
        ):
            if detections:
                print(f"Frame {frame_num:04d}: {len(detections)} hazards detected (Persistence count: {hazard_count})")
    else:
        print("[INFO] Running in Static Image mode...")
        annotated_img, detections = detector.detect_image(str(source_path))
        print(f"Detections found: {len(detections)}")
        for d in detections:
            print(f" - {d['class_name']} ({d['confidence'] * 100:.1f}%) at {d['box']}")

        if args.output:
            cv2.imwrite(args.output, annotated_img)
            print(f"[INFO] Output image saved to {args.output}")


if __name__ == "__main__":
    main()
