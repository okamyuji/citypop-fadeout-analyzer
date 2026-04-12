"""合成音声による解析ロジックのテスト."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from citypop_fadeout_analyzer.analyzer import (
    DEFAULT_SR,
    EndingFeatures,
    classify,
    extract_features,
)


def _sine_wave(
    duration_sec: float,
    freq: float = 220.0,
    sr: int = DEFAULT_SR,
) -> npt.NDArray[np.float64]:
    t = np.arange(int(duration_sec * sr), dtype=np.float64) / sr
    return 0.5 * np.sin(2.0 * np.pi * freq * t)


def _vamp(
    duration_sec: float,
    sr: int = DEFAULT_SR,
) -> npt.NDArray[np.float64]:
    """4 音を繰り返すリフ風の合成信号."""
    notes_hz = [220.0, 293.66, 329.63, 246.94]
    bar_sec = 2.0
    note_sec = bar_sec / len(notes_hz)
    samples_per_note = int(note_sec * sr)
    bar = np.concatenate(
        [0.5 * np.sin(2 * np.pi * f * np.arange(samples_per_note) / sr) for f in notes_hz]
    )
    total_bars = max(1, int(np.ceil(duration_sec / bar_sec)))
    signal = np.tile(bar, total_bars)
    return signal[: int(duration_sec * sr)]


def _apply_fade_out(
    signal: npt.NDArray[np.float64],
    fade_sec: float,
    sr: int = DEFAULT_SR,
) -> npt.NDArray[np.float64]:
    out = signal.copy()
    n_fade = int(fade_sec * sr)
    n_fade = min(n_fade, len(out))
    ramp = np.linspace(1.0, 0.0, n_fade) ** 2.0
    out[-n_fade:] *= ramp
    return out


def _apply_cold_release(
    signal: npt.NDArray[np.float64],
    release_sec: float = 0.3,
    sr: int = DEFAULT_SR,
) -> npt.NDArray[np.float64]:
    out = signal.copy()
    n = int(release_sec * sr)
    n = min(n, len(out))
    ramp = np.exp(-np.linspace(0.0, 8.0, n))
    out[-n:] *= ramp
    return out


@pytest.fixture
def sr() -> int:
    return DEFAULT_SR


def test_fade_out_signal_is_classified_as_fade_out(sr: int) -> None:
    signal = _sine_wave(40.0, sr=sr)
    signal = _apply_fade_out(signal, fade_sec=20.0, sr=sr)

    features = extract_features(signal, sr)
    result = classify(features)
    assert result == "fade_out", f"got {result}, slope={features.tail_slope_db_per_sec}"


def test_cold_ending_signal_is_classified_as_cold(sr: int) -> None:
    signal = _sine_wave(40.0, sr=sr)
    signal = _apply_cold_release(signal, release_sec=0.2, sr=sr)

    features = extract_features(signal, sr)
    result = classify(features)
    assert result == "cold", (
        f"got {result}, rep={features.tail_repetition_score}, drop2s={features.drop_at_2s_db}"
    )


def test_vamp_has_higher_repetition_than_sine(sr: int) -> None:
    vamp_signal = _vamp(40.0, sr=sr)
    vamp_signal = _apply_cold_release(vamp_signal, release_sec=0.2, sr=sr)
    vamp_features = extract_features(vamp_signal, sr)

    sine_signal = _sine_wave(40.0, sr=sr)
    sine_signal = _apply_cold_release(sine_signal, release_sec=0.2, sr=sr)
    sine_features = extract_features(sine_signal, sr)

    assert vamp_features.tail_repetition_score > sine_features.tail_repetition_score


def test_features_are_immutable(sr: int) -> None:
    signal = _sine_wave(10.0, sr=sr)
    features = extract_features(signal, sr)
    with pytest.raises((AttributeError, TypeError)):
        features.reference_level_db = 0.0  # type: ignore[misc]


def test_extract_features_returns_expected_type(sr: int) -> None:
    signal = _sine_wave(8.0, sr=sr)
    features = extract_features(signal, sr)
    assert isinstance(features, EndingFeatures)
    assert features.duration_sec == pytest.approx(8.0, abs=0.05)
