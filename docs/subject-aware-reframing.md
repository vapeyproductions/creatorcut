# Subject-aware vertical reframing

Vertical exports no longer assume the interesting subject is in the geometric center of a
landscape frame. CreatorCut samples frames with OpenCV's frontal-face detector, chooses a prominent
face while preferring continuity with the previous target, and moves the 9:16 crop toward that
subject.

Three controls prevent detector noise from becoming visible camera shake:

- face detection runs every six video frames rather than independently driving every frame;
- the previous subject is held for 30 frames when detection temporarily drops;
- an exponential smoother and a four-percent-per-frame movement cap constrain crop motion.

If OpenCV or the detector data is unavailable, no face is found, or the source does not require a
horizontal crop, export still succeeds through a deterministic center/vertical-fit path. Each
export writes sidecar metadata identifying the actual mode, detector, frame count, detection sample
rate, multi-face samples, and smoothing configuration. That metadata is also stored with the
download feedback event, so downstream quality analysis can distinguish a face-tracked output from
a fallback output.

This version performs face-aware subject tracking, not active-speaker identification. In a
multi-person podcast it favors the most prominent stable face and may not switch to an off-center
speaker at the ideal moment. Active-speaker diarization and a modern face/person detector remain
reasonable later experiments, but the current implementation is local, inspectable, and degrades
without breaking export.
