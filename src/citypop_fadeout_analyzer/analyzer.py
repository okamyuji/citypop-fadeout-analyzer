"""楽曲末尾の振る舞いから終わり方を 3 分類する解析ロジック.

考え方:
    1. フレーム単位の RMS (dB) 系列を作る
    2. 「実効的な終端フレーム」(無音/リリースに落ちる直前) を推定
    3. 終端から遡って一定秒数のテール窓を取り出す
    4. テール窓内の減衰形状 (線形回帰の傾き, T-2s / T-5s / T-10s / T-15s での音量差)
       からフェードアウト性を判定する
    5. テールのクロマ自己相似度から「ヴァンプ (リフ反復)」性を推定する

    fade_out          : テール全体にわたって音量が単調に 8 dB 以上減衰
    cold              : 終端 2 秒前まで本体と同じ音量で鳴っており, リフ反復性も低い
    riff_repeat_cold  : 終端 2 秒前まで本体と同音量で, かつテールのリフ反復性が高い
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import numpy.typing as npt

EndingClass = Literal["fade_out", "cold", "riff_repeat_cold"]
ENDING_CLASSES: tuple[EndingClass, ...] = ("fade_out", "cold", "riff_repeat_cold")

ENDING_LABEL_JA: dict[EndingClass, str] = {
    "fade_out": "Fade-out",
    "cold": "明確終止",
    "riff_repeat_cold": "リフ反復終止",
}

DEFAULT_SR: int = 22_050
FRAME_LENGTH: int = 2048
HOP_LENGTH: int = 512
END_SILENCE_DB: float = -45.0
TAIL_WINDOW_SEC: float = 12.0
REPETITION_WINDOW_SEC: float = 20.0


@dataclass(slots=True, frozen=True)
class EndingFeatures:
    """終わり方分類の説明変数."""

    duration_sec: float
    effective_end_sec: float
    reference_level_db: float
    level_at_2s_db: float
    level_at_5s_db: float
    level_at_10s_db: float
    level_at_15s_db: float
    drop_at_2s_db: float
    drop_at_5s_db: float
    drop_at_10s_db: float
    drop_at_15s_db: float
    tail_slope_db_per_sec: float
    tail_repetition_score: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _compute_rms_db(y: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    rms = np.asarray(rms, dtype=np.float64)
    peak = float(np.max(rms)) if rms.size > 0 else 0.0
    ref = peak if peak > 0.0 else 1.0
    result: npt.NDArray[np.float64] = 20.0 * np.log10(np.maximum(rms, 1e-10) / ref)
    return result


def _find_effective_end_frame(rms_db: npt.NDArray[np.float64]) -> int:
    above = np.where(rms_db > END_SILENCE_DB)[0]
    if above.size == 0:
        return max(0, len(rms_db) - 1)
    return int(above[-1])


def _level_around(
    rms_db: npt.NDArray[np.float64],
    frame_index: int,
    frames_per_sec: float,
    half_window_sec: float = 0.25,
) -> float:
    if frame_index < 0 or rms_db.size == 0:
        return float("-inf")
    half = max(1, int(half_window_sec * frames_per_sec))
    lo = max(0, frame_index - half)
    hi = min(len(rms_db), frame_index + half + 1)
    if lo >= hi:
        return float(rms_db[min(frame_index, len(rms_db) - 1)])
    return float(np.mean(rms_db[lo:hi]))


def _reference_level(rms_db: npt.NDArray[np.float64]) -> float:
    if rms_db.size == 0:
        return 0.0
    lo = int(len(rms_db) * 0.30)
    hi = int(len(rms_db) * 0.70)
    if hi <= lo:
        return float(np.median(rms_db))
    return float(np.percentile(rms_db[lo:hi], 75))


def _tail_slope(
    rms_db: npt.NDArray[np.float64],
    end_frame: int,
    frames_per_sec: float,
) -> float:
    tail_frames = int(TAIL_WINDOW_SEC * frames_per_sec)
    start = max(0, end_frame - tail_frames)
    segment = rms_db[start : end_frame + 1]
    if segment.size < 4:
        return 0.0
    t = np.arange(segment.size, dtype=np.float64) / frames_per_sec
    slope, _ = np.polyfit(t, segment, 1)
    return float(slope)


def _repetition_score(
    y: npt.NDArray[np.floating],
    sr: int,
    end_frame: int,
    frames_per_sec: float,
) -> float:
    """テール区間の反復性をオンセット自己相関で測定する.

    純粋な持続音 (音色変化なし) は 0 に近くなる.
    リフやコード進行のヴァンプが繰り返されると周期的なオンセットが生まれ値が上がる.
    """
    tail_frames = int(REPETITION_WINDOW_SEC * frames_per_sec)
    start_frame = max(0, end_frame - tail_frames)
    y_start = start_frame * HOP_LENGTH
    y_end = min(len(y), end_frame * HOP_LENGTH)
    if y_end - y_start < sr * 4:
        return 0.0
    y_tail = y[y_start:y_end]
    onset_env = librosa.onset.onset_strength(y=y_tail, sr=sr, hop_length=HOP_LENGTH)
    if onset_env.size < 8:
        return 0.0
    onset_env = onset_env - float(np.mean(onset_env))
    norm = float(np.dot(onset_env, onset_env))
    if norm < 1e-9:
        return 0.0
    autocorr = np.correlate(onset_env, onset_env, mode="full")
    autocorr = autocorr[autocorr.size // 2 :]
    autocorr = autocorr / norm
    onset_fps = float(sr) / HOP_LENGTH
    min_lag = max(1, int(0.5 * onset_fps))
    max_lag = min(int(4.0 * onset_fps), len(autocorr) - 1)
    if max_lag <= min_lag:
        return 0.0
    peak = float(np.max(autocorr[min_lag:max_lag]))
    return float(np.clip(peak, 0.0, 1.0))


def extract_features(y: npt.NDArray[np.floating], sr: int) -> EndingFeatures:
    """波形から終わり方分類用の特徴量を抽出する."""
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    peak = float(np.max(np.abs(y)))
    y = (y / peak).astype(np.float64) if peak > 0.0 else y.astype(np.float64)

    rms_db = _compute_rms_db(y)
    frames_per_sec = sr / HOP_LENGTH
    end_frame = _find_effective_end_frame(rms_db)

    reference_level = _reference_level(rms_db)

    def at(seconds_before: float) -> float:
        return _level_around(
            rms_db,
            end_frame - int(seconds_before * frames_per_sec),
            frames_per_sec,
        )

    level_2s = at(2.0)
    level_5s = at(5.0)
    level_10s = at(10.0)
    level_15s = at(15.0)

    return EndingFeatures(
        duration_sec=float(len(y) / sr),
        effective_end_sec=float(end_frame / frames_per_sec),
        reference_level_db=reference_level,
        level_at_2s_db=level_2s,
        level_at_5s_db=level_5s,
        level_at_10s_db=level_10s,
        level_at_15s_db=level_15s,
        drop_at_2s_db=reference_level - level_2s,
        drop_at_5s_db=reference_level - level_5s,
        drop_at_10s_db=reference_level - level_10s,
        drop_at_15s_db=reference_level - level_15s,
        tail_slope_db_per_sec=_tail_slope(rms_db, end_frame, frames_per_sec),
        tail_repetition_score=_repetition_score(y, sr, end_frame, frames_per_sec),
    )


def classify(features: EndingFeatures) -> EndingClass:
    """抽出済み特徴量から終わり方クラスを決定する."""
    gradual_decay = (
        features.drop_at_10s_db >= 8.0
        and features.drop_at_5s_db >= 4.0
        and features.tail_slope_db_per_sec <= -1.0
    )
    if gradual_decay:
        return "fade_out"

    strong_slope = features.tail_slope_db_per_sec <= -0.8
    moderate_drop_at_10s = features.drop_at_10s_db >= 5.0
    if strong_slope and moderate_drop_at_10s:
        return "fade_out"

    still_loud_near_end = features.drop_at_2s_db < 6.0
    if still_loud_near_end and features.tail_repetition_score >= 0.55:
        return "riff_repeat_cold"
    if still_loud_near_end:
        return "cold"

    if features.tail_slope_db_per_sec <= -0.5 and features.drop_at_5s_db >= 3.0:
        return "fade_out"
    return "cold"


def analyze_file(path: Path, sr: int = DEFAULT_SR) -> tuple[EndingFeatures, EndingClass]:
    """音声ファイルを読み込み, 特徴量と分類結果を返す."""
    y, file_sr = librosa.load(str(path), sr=sr, mono=True)
    features = extract_features(y, int(file_sr))
    return features, classify(features)
