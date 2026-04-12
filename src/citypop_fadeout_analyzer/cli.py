"""CLI エントリポイント."""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd

from citypop_fadeout_analyzer.analyzer import (
    ENDING_CLASSES,
    ENDING_LABEL_JA,
    analyze_file,
)
from citypop_fadeout_analyzer.metadata import read_metadata
from citypop_fadeout_analyzer.visualize import render_all

SUPPORTED_EXT = {".mp3", ".flac", ".wav", ".m4a", ".ogg"}


@click.group()
def main() -> None:
    """シティポップ終わり方分類 CLI."""


@main.command("analyze")
@click.option(
    "--audio-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="MP3 などが置かれたディレクトリ",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="分類結果 CSV の書き出し先",
)
def analyze_cmd(audio_dir: Path, out: Path) -> None:
    """指定ディレクトリの全音源を解析して結果 CSV を作成."""
    audio_files = sorted(
        p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    )
    if not audio_files:
        raise click.ClickException(f"no audio files found in {audio_dir}")

    rows: list[dict[str, object]] = []
    with click.progressbar(audio_files, label="Analyzing") as bar:
        for path in bar:
            meta = read_metadata(path)
            try:
                features, ending_class = analyze_file(path)
            except Exception as exc:
                click.echo(f"[skip] {path.name}: {exc}", err=True)
                continue
            row: dict[str, object] = {
                **meta.as_dict(),
                "ending_class": ending_class,
                "ending_label": ENDING_LABEL_JA[ending_class],
                **features.as_dict(),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    click.echo(f"wrote {len(df)} rows to {out}")


@main.command("report")
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
def report_cmd(csv_path: Path) -> None:
    """分類結果の統計サマリを表示."""
    df = pd.read_csv(csv_path)
    total = len(df)
    click.echo(f"total: {total}")
    if total == 0:
        return
    counts = df["ending_class"].value_counts()
    for cls in ENDING_CLASSES:
        n = int(counts.get(cls, 0))
        label = ENDING_LABEL_JA[cls]
        click.echo(f"  {label:>12}: {n:>4} ({n / total * 100:5.1f}%)")
    if "year" in df.columns and df["year"].notna().any():
        click.echo(f"\nyear range: {int(df['year'].min())} - {int(df['year'].max())}")


@main.command("visualize")
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
)
def visualize_cmd(csv_path: Path, out_dir: Path) -> None:
    """分類結果からグラフ一式を生成."""
    df = pd.read_csv(csv_path)
    paths = render_all(df, out_dir)
    for path in paths:
        click.echo(f"wrote {path}")


if __name__ == "__main__":
    main()
