"""Face-aware horizontal crop tracking with a deterministic center fallback."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

FaceBox = tuple[int, int, int, int]
FaceDetector = Callable[[np.ndarray], list[FaceBox]]
REFRAMING_VERSION = "creatorcut_reframing_v1"


def smooth_center(
    previous: float,
    target: float,
    width: int,
    alpha: float = 0.35,
    maximum_step_ratio: float = 0.04,
) -> float:
    """Low-pass a target center and cap jumps that would look like camera shake."""
    proposed = previous + alpha * (target - previous)
    maximum_step = max(2.0, width * maximum_step_ratio)
    delta = max(-maximum_step, min(maximum_step, proposed - previous))
    return previous + delta


def crop_left(width: int, crop_width: int, center_x: float) -> int:
    """Convert a tracked subject center into a valid horizontal crop origin."""
    if crop_width >= width:
        return 0
    return max(0, min(width - crop_width, int(round(center_x - crop_width / 2))))


def _opencv_face_detector() -> tuple[FaceDetector | None, str]:
    try:
        import cv2
    except ImportError:
        return None, "center_fallback_no_opencv"
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        return None, "center_fallback_missing_cascade"

    def detect(pixels: np.ndarray) -> list[FaceBox]:
        height, width = pixels.shape[:2]
        scale = min(1.0, 480.0 / max(width, 1))
        image = pixels
        if scale < 1.0:
            image = cv2.resize(
                pixels,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.12,
            minNeighbors=5,
            minSize=(30, 30),
        )
        inverse = 1.0 / scale
        return [
            (
                int(round(float(x) * inverse)),
                int(round(float(y) * inverse)),
                int(round(float(face_width) * inverse)),
                int(round(float(face_height) * inverse)),
            )
            for x, y, face_width, face_height in faces
        ]

    return detect, "opencv_haar_frontalface_v1"


class SubjectAwareCropper:
    """Track the most stable prominent face while sampling detection work."""

    def __init__(
        self,
        detector: FaceDetector | None = None,
        detector_name: str | None = None,
        detection_interval: int = 6,
        hold_frames: int = 30,
    ) -> None:
        if detection_interval <= 0 or hold_frames < 0:
            raise ValueError("Invalid subject-tracker timing")
        if detector is None:
            detector, discovered_name = _opencv_face_detector()
            detector_name = detector_name or discovered_name
        self.detector = detector
        self.detector_name = detector_name or "custom_face_detector"
        self.detection_interval = detection_interval
        self.hold_frames = hold_frames
        self.center: float | None = None
        self.target: float | None = None
        self.frames_since_face = hold_frames + 1
        self.frame_count = 0
        self.sampled_frame_count = 0
        self.face_detected_frame_count = 0
        self.multi_face_frame_count = 0

    def _choose_face(self, faces: list[FaceBox], width: int) -> FaceBox:
        previous = self.target if self.target is not None else width / 2

        def score(face: FaceBox) -> float:
            x, _y, face_width, face_height = face
            area = max(1, face_width * face_height)
            center = x + face_width / 2
            distance_penalty = 1.0 - 0.45 * min(1.0, abs(center - previous) / width)
            return area * distance_penalty

        return max(faces, key=score)

    def center_for_frame(self, pixels: np.ndarray, frame_index: int) -> float:
        """Return a smoothed subject center for one RGB frame."""
        _height, width = pixels.shape[:2]
        midpoint = width / 2
        face_found = False
        if self.detector is not None and frame_index % self.detection_interval == 0:
            self.sampled_frame_count += 1
            faces = self.detector(pixels)
            if faces:
                self.face_detected_frame_count += 1
                self.multi_face_frame_count += int(len(faces) > 1)
                face = self._choose_face(faces, width)
                self.target = face[0] + face[2] / 2
                self.frames_since_face = 0
                face_found = True
        if not face_found:
            self.frames_since_face += 1
            if self.frames_since_face > self.hold_frames:
                self.target = midpoint
        target = self.target if self.target is not None else midpoint
        if self.center is None:
            self.center = midpoint
        self.center = smooth_center(
            self.center,
            target,
            width,
            alpha=0.35 if face_found else 0.08,
        )
        self.frame_count += 1
        return self.center

    def metadata(self, horizontal_crop: bool = True) -> dict[str, Any]:
        """Describe whether subject tracking actually influenced the export."""
        detected = self.face_detected_frame_count
        mode = (
            "subject_aware_face_tracking"
            if horizontal_crop and self.detector is not None and detected > 0
            else "center_crop_fallback"
            if horizontal_crop
            else "vertical_source_fit"
        )
        return {
            "schema": REFRAMING_VERSION,
            "mode": mode,
            "detector": self.detector_name,
            "frame_count": self.frame_count,
            "sampled_frame_count": self.sampled_frame_count,
            "face_detected_frame_count": detected,
            "multi_face_frame_count": self.multi_face_frame_count,
            "sample_detection_rate": round(
                detected / self.sampled_frame_count, 4
            )
            if self.sampled_frame_count
            else None,
            "smoothing": {
                "detection_interval_frames": self.detection_interval,
                "hold_frames": self.hold_frames,
                "maximum_step_ratio": 0.04,
            },
        }
