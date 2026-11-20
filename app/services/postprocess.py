"""Deterministic post-processing. Pure functions: no I/O, no model, no paraphrasing."""

from collections.abc import Iterable

from app.providers.base import RawSegment
from app.schemas.transcription import Segment

_TIMESTAMP_DECIMALS = 3  # milliseconds; removes float noise without inventing precision


def normalize_whitespace(text: str) -> str:
    """Trim and collapse all runs of whitespace to single spaces. Changes nothing else."""
    return " ".join(text.split())


def build_segments(raw_segments: Iterable[RawSegment]) -> list[Segment]:
    """Normalize text, drop segments that are empty afterwards, and order by start time."""
    segments: list[Segment] = []
    for raw in raw_segments:
        text = normalize_whitespace(raw.text)
        if not text:
            continue
        start = max(0.0, round(raw.start, _TIMESTAMP_DECIMALS))
        end = max(start, round(raw.end, _TIMESTAMP_DECIMALS))
        segments.append(Segment(start=start, end=end, text=text))
    # sorted() is stable, so segments with equal starts keep the model's order.
    return sorted(segments, key=lambda segment: segment.start)


def assemble_transcript(segments: Iterable[Segment]) -> str:
    return " ".join(segment.text for segment in segments)


def count_words(text: str) -> int:
    """Whitespace-delimited count. Not a linguistic tokenizer (undercounts e.g. Chinese)."""
    return len(text.split())
