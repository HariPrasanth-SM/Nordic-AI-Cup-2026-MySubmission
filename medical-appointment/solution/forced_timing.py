"""Optional Exp11: replace timing only, preserving Exp10 word selections/text.
Reject token/text mismatch instead of silently changing the evidence occurrence.
"""
import bisect
import logging
import math
import os
import tempfile
from dataclasses import replace
from solution.evidence_experiment import normalize
from solution.experiment_trace import event
from solution.types import Span

logger = logging.getLogger(__name__)


def map_alignment(words, aligned):
    original = [normalize(w.text) for w in words]
    aligned_text = [normalize(str(w.text)) for w in aligned]
    if ''.join(original) != ''.join(aligned_text) or not ''.join(original):
        raise ValueError('Aligner changed normalized transcript; refusing to map word indices')
    records, starts, ends = [], [], []
    cursor = 0
    for text, w in zip(aligned_text, aligned):
        if not text:
            continue
        a, b = float(w.start_time), float(w.end_time)
        if not math.isfinite(a+b) or a < 0 or b < a:
            raise ValueError('Invalid forced-alignment timestamp')
        starts.append(cursor); cursor += len(text); ends.append(cursor)
        records.append((a,b))
    if any(records[i][0] < records[i-1][0] for i in range(1,len(records))):
        raise ValueError('Non-monotonic forced alignment')
    mapped, cursor = [], 0
    for w, text in zip(words, original):
        if not text:
            mapped.append(w); continue
        left = bisect.bisect_right(ends, cursor)
        right = bisect.bisect_left(starts, cursor+len(text))-1
        mapped.append(replace(w,start=records[left][0],end=records[right][1]))
        cursor += len(text)
    return mapped


class ForcedTiming:
    def __init__(self):
        import torch
        from qwen_asr import Qwen3ForcedAligner
        self.model = Qwen3ForcedAligner.from_pretrained(
            os.environ.get('MEDICAL_ALIGNER_MODEL','Qwen/Qwen3-ForcedAligner-0.6B'),
            dtype=torch.bfloat16, device_map='cuda:0')

    def refine(self, audio_bytes, segments, results):
        words = [w for s in segments for w in s.words]
        if not words:
            return results
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp3') as f:
                f.write(audio_bytes); f.flush()
                aligned = self.model.align(audio=f.name,
                    text=' '.join(w.text for w in words),language='English')[0]
            mapped = map_alignment(words, aligned)
            output = []
            for r in results:
                if r.span is None:
                    output.append(r); continue
                # Exp10 and baseline both return word endpoints. Reject any
                # unexpected endpoint rather than silently choose another word.
                left = min(range(len(words)),key=lambda i:abs(words[i].start-r.span.start))
                right = min(range(left,len(words)),key=lambda i:abs(words[i].end-r.span.end))
                if (abs(words[left].start-r.span.start) > .025 or
                    abs(words[right].end-r.span.end) > .025 or
                    mapped[right].end <= mapped[left].start):
                    output.append(r); continue
                output.append(replace(r,span=Span(mapped[left].start,mapped[right].end)))
            event('forced_alignment', words=[{'text':w.text,'start':w.start,'end':w.end}
                                            for w in mapped])
            return output
        except Exception as exc:
            logger.exception('Forced alignment failed; preserving Exp10 timing')
            event('forced_alignment_failed', error=str(exc))
            return results
