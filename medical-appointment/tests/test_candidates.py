from solution.candidates import build_candidate_windows
from solution.types import Segment, Word


def _segment(start: float, end: float, text: str) -> Segment:
    words = []
    n = len(text.split())
    step = (end - start) / max(1, n)
    for i, token in enumerate(text.split()):
        words.append(Word(text=token, start=start + i * step, end=start + (i + 1) * step))
    return Segment(start=start, end=end, text=text, words=words)


def test_empty_segments_gives_empty_windows():
    assert build_candidate_windows([], merge_sizes=(1, 2, 3)) == []


def test_size_1_gives_one_window_per_segment():
    segments = [_segment(0, 2, "hello there"), _segment(2, 4, "how are"), _segment(4, 6, "you doing")]
    windows = build_candidate_windows(segments, merge_sizes=(1,))
    assert len(windows) == 3
    assert [w.text for w in windows] == ["hello there", "how are", "you doing"]


def test_size_2_gives_adjacent_pairs_only():
    segments = [_segment(0, 2, "a"), _segment(2, 4, "b"), _segment(4, 6, "c")]
    windows = build_candidate_windows(segments, merge_sizes=(2,))
    assert len(windows) == 2   # (a,b) and (b,c); no wraparound, no (a,c)
    assert windows[0].text == "a b"
    assert windows[1].text == "b c"
    assert windows[0].start == 0
    assert windows[0].end == 4


def test_merged_window_concatenates_words_in_order():
    segments = [_segment(0, 2, "a b"), _segment(2, 4, "c d")]
    windows = build_candidate_windows(segments, merge_sizes=(2,))
    assert len(windows) == 1
    assert [w.text for w in windows[0].words] == ["a", "b", "c", "d"]
    assert windows[0].start == 0
    assert windows[0].end == 4


def test_size_larger_than_segment_count_yields_no_window_of_that_size():
    segments = [_segment(0, 2, "a"), _segment(2, 4, "b")]
    windows = build_candidate_windows(segments, merge_sizes=(1, 5))
    # size-5 windows are impossible with only 2 segments; should not crash,
    # and should not silently produce a window shorter than requested
    assert len(windows) == 2   # only the two size-1 windows
