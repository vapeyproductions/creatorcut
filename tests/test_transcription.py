from types import SimpleNamespace

from creatorcut.transcription import transcribe_video


class FakeTranscriber:
    def __init__(self) -> None:
        self.options = None

    def transcribe(self, _path: str, **options):
        self.options = options
        info = SimpleNamespace(
            language="en",
            language_probability=1.0,
            duration=30.0,
            duration_after_vad=30.0,
        )
        return iter([]), info


def test_transcribe_video_can_disable_vad(tmp_path):
    transcriber = FakeTranscriber()
    result = transcribe_video(
        transcriber,
        "video_010",
        tmp_path / "video_010.mp4",
        tmp_path / "video_010.json",
        "small.en",
        "int8",
        batch_size=1,
        vad_filter=False,
    )

    assert transcriber.options["vad_filter"] is False
    assert "vad_parameters" not in transcriber.options
    assert result["model"]["vad_filter"] is False


def test_transcribe_video_records_batched_vad_settings(tmp_path):
    transcriber = FakeTranscriber()
    result = transcribe_video(
        transcriber,
        "video_004",
        tmp_path / "video_004.mp4",
        tmp_path / "video_004.json",
        "small.en",
        "int8",
        batch_size=4,
        vad_filter=True,
    )

    assert transcriber.options["batch_size"] == 4
    assert transcriber.options["vad_parameters"] == {"min_silence_duration_ms": 500}
    assert result["model"]["inference_mode"] == "batched"
