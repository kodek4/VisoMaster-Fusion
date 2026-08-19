"""Tests for the compact source/playback FPS display."""

from __future__ import annotations

from types import SimpleNamespace

from app.processors.video_utils.media_pipeline import MediaPipeline


class _TextField:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, text: str) -> None:
        self.text = text


def _pipeline(source_fps: float = 24.0) -> tuple[MediaPipeline, _TextField]:
    field = _TextField()
    pipeline = MediaPipeline.__new__(MediaPipeline)
    pipeline.vp = SimpleNamespace(
        recording_source_fps=source_fps,
        fps=source_fps,
        media_capture=None,
    )
    pipeline.main_window = SimpleNamespace(videoFpsLineEdit=field)
    pipeline._fps_window_start_sec = 0.0
    pipeline._fps_window_frames = 0
    pipeline._display_fps_ema = 0.0
    return pipeline, field


def test_fps_display_starts_with_source_and_zero_playback() -> None:
    pipeline, field = _pipeline(23.976)

    pipeline._reset_playback_fps_display(now_sec=1.0)

    assert field.text == "24/0"


def test_fps_display_reports_rolling_display_throughput() -> None:
    pipeline, field = _pipeline(24.0)
    pipeline._reset_playback_fps_display(now_sec=1.0)

    for frame_number in range(1, 19):
        pipeline._update_playback_fps_display(
            now_sec=1.0 + frame_number / 18.0
        )

    assert field.text == "24/18"


def test_fps_display_does_not_query_the_live_capture() -> None:
    class CaptureMustNotBeRead:
        def get(self, _property):
            raise AssertionError("live capture was queried")

    pipeline, field = _pipeline(30.0)
    pipeline.vp.media_capture = CaptureMustNotBeRead()

    pipeline._reset_playback_fps_display(now_sec=1.0)

    assert field.text == "30/0"
