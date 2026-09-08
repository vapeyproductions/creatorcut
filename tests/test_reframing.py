import numpy as np
import pytest

from creatorcut.reframing import SubjectAwareCropper, crop_left, smooth_center


def test_crop_left_follows_subject_without_leaving_frame():
    assert crop_left(320, 100, 50) == 0
    assert crop_left(320, 100, 170) == 120
    assert crop_left(320, 100, 300) == 220


def test_smoothing_caps_large_face_detector_jumps():
    assert smooth_center(160, 300, 320) == pytest.approx(172.8)


def test_tracker_prefers_stable_face_and_reports_subject_mode():
    detections = iter(
        [
            [(30, 20, 60, 60), (220, 20, 80, 80)],
            [(35, 20, 60, 60), (210, 20, 80, 80)],
        ]
    )
    tracker = SubjectAwareCropper(
        detector=lambda _pixels: next(detections),
        detector_name="test_detector",
        detection_interval=1,
    )
    pixels = np.zeros((180, 320, 3), dtype=np.uint8)

    first = tracker.center_for_frame(pixels, 0)
    second = tracker.center_for_frame(pixels, 1)
    metadata = tracker.metadata()

    assert first > 160
    assert second > first
    assert metadata["mode"] == "subject_aware_face_tracking"
    assert metadata["face_detected_frame_count"] == 2
    assert metadata["multi_face_frame_count"] == 2
