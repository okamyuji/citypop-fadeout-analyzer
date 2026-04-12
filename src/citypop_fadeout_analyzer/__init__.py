"""シティポップの終わり方 (Fade-out / 明確終止 / リフ反復終止) を分類するパッケージ."""

from citypop_fadeout_analyzer.analyzer import (
    ENDING_CLASSES,
    ENDING_LABEL_JA,
    EndingClass,
    EndingFeatures,
    analyze_file,
    classify,
    extract_features,
)
from citypop_fadeout_analyzer.metadata import TrackMetadata, read_metadata

__all__ = [
    "ENDING_CLASSES",
    "ENDING_LABEL_JA",
    "EndingClass",
    "EndingFeatures",
    "TrackMetadata",
    "analyze_file",
    "classify",
    "extract_features",
    "read_metadata",
]

__version__ = "0.1.0"
