"""音源ファイルからメタデータ (アーティスト/タイトル/年/アルバム) を取り出す.

ID3/Vorbis/MP4 タグを mutagen 経由で読み, 欠損した場合はファイル名から復元を試みる.
ファイル名のフォールバック規則:
    <year>__<artist>__<title>.mp3
    <year>_<artist>_<title>.mp3
    <artist> - <title>.mp3
    <title>.mp3
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from mutagen import File as MutagenFile  # type: ignore[attr-defined]

_YEAR_RE = re.compile(r"(?P<year>(?:19|20)\d{2})")


@dataclass(slots=True, frozen=True)
class TrackMetadata:
    filename: str
    artist: str
    title: str
    album: str
    year: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "artist": self.artist,
            "title": self.title,
            "album": self.album,
            "year": self.year,
        }


def _first_text(tag_value: object) -> str:
    if tag_value is None:
        return ""
    if isinstance(tag_value, list):
        if not tag_value:
            return ""
        return str(tag_value[0])
    return str(tag_value)


def _parse_year(value: str) -> int | None:
    if not value:
        return None
    match = _YEAR_RE.search(value)
    if not match:
        return None
    try:
        return int(match.group("year"))
    except ValueError:
        return None


def _extract_tags(path: Path) -> dict[str, str]:
    try:
        tag_file = MutagenFile(path, easy=True)
    except Exception:
        return {}
    if tag_file is None:
        return {}
    tags = getattr(tag_file, "tags", None)
    if tags is None:
        return {}
    result: dict[str, str] = {}
    for key in ("artist", "title", "album", "date", "year", "originaldate"):
        if key in tags:
            value = _first_text(tags[key])
            if value:
                result[key] = value
    return result


def _from_filename(stem: str) -> tuple[str, str, str, int | None]:
    artist = ""
    title = ""
    album = ""
    year: int | None = None

    for sep in ("__", "_"):
        parts = stem.split(sep)
        if len(parts) >= 3 and _parse_year(parts[0]) is not None:
            year = _parse_year(parts[0])
            artist = parts[1].strip()
            title = sep.join(parts[2:]).strip()
            return artist, title, album, year

    if " - " in stem:
        head, tail = stem.split(" - ", 1)
        artist = head.strip()
        title = tail.strip()
        return artist, title, album, year

    return artist, stem, album, year


def read_metadata(path: Path) -> TrackMetadata:
    tags = _extract_tags(path)
    fn_artist, fn_title, fn_album, fn_year = _from_filename(path.stem)

    year = (
        _parse_year(tags.get("date", ""))
        or _parse_year(tags.get("year", ""))
        or _parse_year(tags.get("originaldate", ""))
        or fn_year
    )
    artist = tags.get("artist") or fn_artist
    title = tags.get("title") or fn_title
    album = tags.get("album") or fn_album

    return TrackMetadata(
        filename=path.name,
        artist=artist,
        title=title,
        album=album,
        year=year,
    )
