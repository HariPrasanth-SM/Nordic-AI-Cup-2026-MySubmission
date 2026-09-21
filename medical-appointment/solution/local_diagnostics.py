"""Offline diagnostics. No inference module imports ground truth or this module."""
import csv
import json
import statistics
from collections import Counter,defaultdict
from pathlib import Path


def iou(a,b):
    if a is None or b is None: return 0.0
    overlap=max(0,min(a[1],b[1])-max(a[0],b[0]))
    union=max(a[1],b[1])-min(a[0],b[0])
    return overlap/union if union>0 else 0.0


def as_pair(span):
    if span is None: return None
    if isinstance(span,dict):return (span['start'],span['end'])
    return tuple(span)


def word_oracle(words,gold):
    """Exact maximum tIoU over ALL contiguous word endpoint pairs, O(n²)."""
    best, pair=0.0,None
    for i,w in enumerate(words):
        if w['start']>=gold[1]: continue
        for v in words[i:]:
            if v['end']<=gold[0]:continue
            candidate=(w['start'],v['end'])
            score=iou(gold,candidate)
            if score>best: best,pair=score,candidate
    return best,pair


def text_at(words,span):
    if not span:return ''
    return ' '.join(w['text'] for w in words if span[0] <= (w['start']+w['end'])/2 <= span[1])


def analyze(run_dir,csv_path):
    run_dir=Path(run_dir)
    with open(csv_path) as f:
        rows=list(csv.DictReader(f))
    responses={r['audio_filename']:r for r in map(json.loads,(run_dir/'predictions.jsonl').read_text().splitlines())}
    trace_path=run_dir/'trace.jsonl'
    traces={r['audio_filename']:r for r in map(json.loads,trace_path.read_text().splitlines())} if trace_path.exists() else {}
    details=[]
    for row in rows:
        filename='conversation_'+row['transcript_id']+'.mp3'
        response=responses.get(filename,{})
        qs=response.get('questions',[])
        # Match by question ID logged by client; supports duplicate question text.
        ids=response.get('question_ids',[])
        idx=ids.index(row['question_id']) if row['question_id'] in ids else None
        pred=response['predictions'][idx] if idx is not None else -1
        span=as_pair(response['spans'][idx]) if idx is not None else None
        label=int(row['label']);gold=(float(row['evidence_start']),float(row['evidence_end'])) if label else None
        t=traces.get(filename,{})
        words=[w for s in t.get('segments',[]) for w in s['words']]
        tiou=iou(gold,span) if label else None
        detail=dict(question_id=row['question_id'],conversation=row['transcript_id'],
            question=row['question'],question_type=row['question_type'],label=label,
            prediction=pred,correct=int(pred==label),gold_start=gold[0] if gold else None,
            gold_end=gold[1] if gold else None,pred_start=span[0] if span else None,
            pred_end=span[1] if span else None,tiou=tiou,
            gold_text=text_at(words,gold),pred_text=text_at(words,span),
            latency_ms=response.get('latency_ms'),failure=response.get('error','not sent') if pred==-1 else '')
        if label:
            detail['start_error_s']=span[0]-gold[0] if span else None
            detail['end_error_s']=span[1]-gold[1] if span else None
            detail['duration_ratio']=(span[1]-span[0])/(gold[1]-gold[0]) if span else None
            detail['word_oracle'],_=word_oracle(words,gold) if words else (None,None)
            candidates=[];baseline=None;aligned_words=None;pre=None
            trace_index=t.get('questions',[]).index(row['question']) if row['question'] in t.get('questions',[]) else None
            for event in t.get('events',[]):
                if event['stage']=='candidates' and event['index']==trace_index:
                    candidates.extend((c['start'],c['end']) for c in event['candidates'])
                if event['stage']=='baseline' and trace_index is not None:
                    baseline=as_pair(event['results'][trace_index]['span'])
                if event['stage']=='pre_alignment' and trace_index is not None:
                    pre=as_pair(event['results'][trace_index]['span'])
                if event['stage']=='forced_alignment':aligned_words=event['words']
            detail['candidate_oracle']=max([iou(gold,c) for c in candidates],default=None)
            detail['baseline_tiou']=iou(gold,baseline) if baseline else None
            detail['pre_alignment_tiou']=iou(gold,pre) if pre else None
            detail['aligned_word_oracle']=word_oracle(aligned_words,gold)[0] if aligned_words else None
            # Categories describe interval geometry, not clinical semantic truth.
            if pred==-1:kind='request_failure'
            elif pred==0:kind='false_negative'
            elif span is None:kind='missing_span'
            elif tiou==0:kind='nonoverlap_review_occurrence'
            elif tiou>=.9:kind='strong'
            elif detail['duration_ratio']>1.5:kind='too_wide'
            elif detail['duration_ratio']<.67:kind='too_narrow'
            else:kind='boundary_shift_or_partial'
            detail['error_bucket']=kind
        else:detail['error_bucket']='correct_no' if pred==0 else ('request_failure' if pred==-1 else 'false_positive')
        details.append(detail)
    keys=list(dict.fromkeys(k for d in details for k in d))
    with (run_dir/'diagnosis.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(details)
    positives=[d for d in details if d['label']]
    mean=lambda xs:statistics.mean(xs) if xs else None
    acc=mean([d['correct'] for d in details]); mt=mean([d['tiou'] for d in positives])
    summary=dict(questions=len(details),accuracy=acc,mean_tiou=mt,score=.4*acc+.6*mt,
        target_tiou_for_095=(.95-.4*acc)/.6,
        failures=sum(d['prediction']==-1 for d in details),
        buckets=dict(Counter(d['error_bucket'] for d in details)),
        trace_conversations=len(traces),response_conversations=len(responses))
    for key in ['word_oracle','candidate_oracle','aligned_word_oracle','baseline_tiou','pre_alignment_tiou']:
        vals=[d[key] for d in positives if d.get(key) is not None]
        summary[key+'_observed_mean']=mean(vals);summary[key+'_coverage']=len(vals)
    summary['stage_counts']=dict(Counter(e['stage'] for t in traces.values() for e in t.get('events',[])))
    (run_dir/'diagnosis.json').write_text(json.dumps(summary,indent=2))
    lines=['# Run diagnosis','',f'Accuracy **{acc:.4f}**; tIoU **{mt:.4f}**; score **{summary["score"]:.4f}**.',
        f'All {len(details)} gold questions are included; unsent/failed requests count wrong.',
        '', '## Error categories','', '| Category | Count |','|---|---:|']
    lines += [f'| {k} | {v} |' for k,v in summary['buckets'].items()]
    lines += ['', '## Oracles and paired diagnostics','',
        'Oracles use gold and are ceilings, not deployable predictions. Means below cover only rows with the required trace.',
        'Candidate oracle covers LLM-proposed hypotheses only; it is not an exhaustive retrieval oracle.',
        '', '| Metric | Mean | Positive rows covered |','|---|---:|---:|']
    for key in ['word_oracle','candidate_oracle','aligned_word_oracle','baseline_tiou','pre_alignment_tiou']:
        lines.append(f'| {key} | {summary[key+"_observed_mean"]} | {summary[key+"_coverage"]} |')
    lines += ['', '## What to change next','',
        '- Many nonoverlapping intervals: read gold_text versus pred_text; inspect occurrence choice. Timing alone cannot repair this.',
        '- Candidate oracle high, selected tIoU low: selection/prompt convention is the bottleneck.',
        '- Word oracle high, candidate oracle low: broaden source hypotheses and boundary choices.',
        '- Aligned word oracle higher but actual tIoU lower: better acoustic timestamps disagree with annotation convention; reject the alignment change.',
        '- Many false negatives: inspect ASR numbers/negations and classification threshold before localization.',
        '- Missing trace/stage fallbacks: fix execution before judging model quality.',
        '', '## Twenty worst positive spans','']
    for d in sorted(positives,key=lambda d:d['tiou'])[:20]:
        lines += [f'### {d["question_id"]}: tIoU {d["tiou"]:.3f}',d['question'],
            f'Gold {d["gold_start"]}–{d["gold_end"]}: {d["gold_text"]}',
            f'Prediction {d["pred_start"]}–{d["pred_end"]}: {d["pred_text"]}','']
    (run_dir/'diagnosis.md').write_text('\n'.join(lines))
    return summary
