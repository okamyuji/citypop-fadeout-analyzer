"""メタデータ読み取りのテスト."""

from __future__ import annotations

from citypop_fadeout_analyzer.metadata import TrackMetadata, _from_filename


def test_filename_with_year_and_underscores() -> None:
    artist, title, _album, year = _from_filename("1980__Tatsuro Yamashita__Ride On Time")
    assert year == 1980
    assert artist == "Tatsuro Yamashita"
    assert title == "Ride On Time"


def test_filename_with_dash_separator() -> None:
    artist, title, _album, year = _from_filename("Mariya Takeuchi - Plastic Love")
    assert artist == "Mariya Takeuchi"
    assert title == "Plastic Love"
    assert year is None


def test_filename_title_only() -> None:
    _artist, title, _album, year = _from_filename("unknown_track")
    assert title == "unknown_track"
    assert year is None


def test_track_metadata_as_dict() -> None:
    meta = TrackMetadata(filename="a.mp3", artist="A", title="T", album="AL", year=1984)
    d = meta.as_dict()
    assert d["artist"] == "A"
    assert d["year"] == 1984
