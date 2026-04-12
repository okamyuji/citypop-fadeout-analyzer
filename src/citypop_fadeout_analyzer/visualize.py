"""分類結果の可視化."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from citypop_fadeout_analyzer.analyzer import ENDING_CLASSES, ENDING_LABEL_JA

PALETTE: dict[str, str] = {
    "fade_out": "#E85A4F",
    "cold": "#3A6EA5",
    "riff_repeat_cold": "#F4A261",
}


def _apply_style() -> None:
    sns.set_theme(style="whitegrid", font="Hiragino Sans")
    plt.rcParams["axes.unicode_minus"] = False


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ending_label"] = df["ending_class"].map(ENDING_LABEL_JA)
    return df


def plot_class_distribution(df: pd.DataFrame, out_path: Path) -> None:
    _apply_style()
    df = _ordered(df)
    counts = (
        df["ending_class"]
        .value_counts()
        .reindex(list(ENDING_CLASSES), fill_value=0)
        .rename(index=ENDING_LABEL_JA)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        counts.index.tolist(),
        counts.values.tolist(),
        color=[PALETTE[c] for c in ENDING_CLASSES],
        edgecolor="#222",
        linewidth=0.6,
    )
    ax.set_title(f"シティポップ {len(df)} 曲の終わり方分類", fontsize=14)
    ax.set_ylabel("曲数")
    max_value = int(counts.to_numpy().max())
    ax.set_ylim(0, max(max_value + 5, 10))
    for bar, value in zip(bars, counts.to_numpy(), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{value} ({value / len(df) * 100:.1f}%)",
            ha="center",
            fontsize=11,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_year_trend(df: pd.DataFrame, out_path: Path) -> None:
    _apply_style()
    df = _ordered(df)
    df = df.dropna(subset=["year"])
    if df.empty:
        return
    df["year_bucket"] = (df["year"] // 2) * 2

    grouped = (
        df.groupby(["year_bucket", "ending_class"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(ENDING_CLASSES), fill_value=0)
    )
    share = grouped.div(grouped.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    bottoms = pd.Series(0.0, index=share.index)
    for cls in ENDING_CLASSES:
        ax.bar(
            share.index.astype(str),
            share[cls].values.tolist(),
            bottom=bottoms.values.tolist(),
            color=PALETTE[cls],
            edgecolor="#222",
            linewidth=0.4,
            label=ENDING_LABEL_JA[cls],
        )
        bottoms = bottoms + share[cls]

    ax.set_title("年代別の終わり方構成比 (2 年バケット)", fontsize=14)
    ax.set_xlabel("リリース年")
    ax.set_ylabel("構成比")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_scatter(df: pd.DataFrame, out_path: Path) -> None:
    _apply_style()
    df = _ordered(df)
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in ENDING_CLASSES:
        subset = df[df["ending_class"] == cls]
        ax.scatter(
            subset["drop_at_10s_db"],
            subset["tail_slope_db_per_sec"],
            color=PALETTE[cls],
            label=ENDING_LABEL_JA[cls],
            s=70,
            alpha=0.85,
            edgecolor="#222",
            linewidth=0.5,
        )
    ax.axvline(8.0, color="#888", linestyle="--", linewidth=0.8)
    ax.axhline(-1.0, color="#888", linestyle="--", linewidth=0.8)
    ax.set_xlabel("終端 10 秒前の音量差 (dB)")
    ax.set_ylabel("テール傾き (dB/秒)")
    ax.set_title("分類判定の特徴量空間", fontsize=14)
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_artist_breakdown(df: pd.DataFrame, out_path: Path, min_songs: int = 2) -> None:
    _apply_style()
    df = _ordered(df)
    df = df.dropna(subset=["artist"])
    if df.empty:
        return
    counts = df["artist"].value_counts()
    qualifying = counts[counts >= min_songs].index.tolist()
    if not qualifying:
        return
    subset = df[df["artist"].isin(qualifying)]
    grouped = (
        subset.groupby(["artist", "ending_class"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(ENDING_CLASSES), fill_value=0)
        .loc[qualifying]
    )

    fig, ax = plt.subplots(figsize=(9, max(4, len(qualifying) * 0.55)))
    bottoms = pd.Series(0.0, index=grouped.index)
    for cls in ENDING_CLASSES:
        ax.barh(
            grouped.index.tolist(),
            grouped[cls].values.tolist(),
            left=bottoms.values.tolist(),
            color=PALETTE[cls],
            edgecolor="#222",
            linewidth=0.4,
            label=ENDING_LABEL_JA[cls],
        )
        bottoms = bottoms + grouped[cls]
    ax.invert_yaxis()
    ax.set_title(f"アーティスト別終わり方 ({min_songs} 曲以上)", fontsize=14)
    ax.set_xlabel("曲数")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


Renderer = Callable[[pd.DataFrame, Path], None]


def render_all(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    targets: list[tuple[str, Renderer]] = [
        ("class_distribution.png", plot_class_distribution),
        ("year_trend.png", plot_year_trend),
        ("feature_scatter.png", plot_feature_scatter),
        ("artist_breakdown.png", plot_artist_breakdown),
    ]
    written: list[Path] = []
    for filename, fn in targets:
        path = out_dir / filename
        fn(df, path)
        written.append(path)
    return written
