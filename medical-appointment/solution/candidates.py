"""Build coarse candidate windows from ASR segments.

Pure logic, no model dependency -- this is the cheap first stage. It never
decides an answer or a final span; it just proposes places worth checking.
Fine-grained tightening down to the annotated spans' actual length
(~2.9s median) happens later, in span.py, only inside whichever coarse
windows retrieval says are worth searching.
"""

from __future__ import annotations

from typing import List, Sequence

from solution.types import Segment, Window


def build_candidate_windows(
    segments: Sequence[Segment],
    merge_sizes: Sequence[int] = (1, 2, 3),
) -> List[Window]:
    """Every window formed by merging 1, 2, ... or N adjacent segments.

    For merge_sizes=(1, 2, 3) and segments [s0, s1, s2, s3]:
        size 1: [s0], [s1], [s2], [s3]
        size 2: [s0,s1], [s1,s2], [s2,s3]
        size 3: [s0,s1,s2], [s1,s2,s3]

    Deliberately not deduplicated across sizes -- a short conversation with
    very few segments may have a size-1 window identical to a size-2 window
    at the start/end, and that's harmless: retrieval scores it once either
    way and duplicate near-identical candidates don't change the argmax.
    """

    if not segments:
        return []

    windows: List[Window] = []
    n = len(segments)

    for size in sorted(set(merge_sizes)):
        if size < 1:
            continue
        for start_idx in range(0, max(0, n - size + 1)):
            chunk = segments[start_idx : start_idx + size]
            if not chunk:
                continue
            windows.append(_merge_segments(chunk, tuple(range(start_idx, start_idx + len(chunk)))))

    return windows


def _merge_segments(chunk: Sequence[Segment], segment_indices: tuple) -> Window:
    words = [w for seg in chunk for w in seg.words]
    text = " ".join(seg.text.strip() for seg in chunk if seg.text.strip())
    return Window(
        start=chunk[0].start,
        end=chunk[-1].end,
        text=text,
        words=words,
        segment_indices=segment_indices,
    )
