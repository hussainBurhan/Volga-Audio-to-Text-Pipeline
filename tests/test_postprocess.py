from app.providers.base import RawSegment
from app.services.postprocess import (
    assemble_transcript,
    build_segments,
    count_words,
    normalize_whitespace,
)


def test_normalize_trims_and_collapses_whitespace():
    assert normalize_whitespace("  Hello \t world\n\nagain  ") == "Hello world again"


def test_normalize_changes_nothing_but_whitespace():
    # Casing, punctuation, fillers and apparent mistakes are evidence; leave them alone.
    text = "Um, we we NEED to  finalise the deployment by friday!!"
    assert normalize_whitespace(text) == "Um, we we NEED to finalise the deployment by friday!!"
    assert normalize_whitespace(text).split() == text.split()


def test_segments_are_sorted_by_start_time():
    raw = [RawSegment(4.0, 6.0, "third"), RawSegment(0.0, 2.0, "first"), RawSegment(2.0, 4.0, "second")]
    assert [s.text for s in build_segments(raw)] == ["first", "second", "third"]


def test_segments_preserve_offsets_and_strip_float_noise():
    segments = build_segments([RawSegment(1.2300000000000002, 4.0600000000000005, " hi ")])
    assert (segments[0].start, segments[0].end, segments[0].text) == (1.23, 4.06, "hi")


def test_whitespace_only_segments_are_dropped():
    segments = build_segments([RawSegment(0.0, 1.0, "   "), RawSegment(1.0, 2.0, "kept")])
    assert [s.text for s in segments] == ["kept"]


def test_timestamps_always_satisfy_zero_le_start_le_end():
    segments = build_segments([RawSegment(-0.02, 1.0, "a"), RawSegment(3.0, 2.5, "b")])
    assert all(0 <= s.start <= s.end for s in segments)


def test_transcript_is_segments_joined_with_single_spaces():
    segments = build_segments([RawSegment(0.0, 1.0, " Hello   world."), RawSegment(1.0, 2.0, " Bye. ")])
    assert assemble_transcript(segments) == "Hello world. Bye."


def test_no_segments_gives_empty_transcript_and_zero_words():
    assert assemble_transcript([]) == ""
    assert count_words("") == 0


def test_word_count_is_whitespace_delimited():
    assert count_words("We need to finalize the deployment by Friday.") == 8
    assert count_words("state-of-the-art  isn't") == 2
