"""Post-run execution gate and full-denominator localization metrics."""
import csv,json,statistics
from collections import Counter
from pathlib import Path
from solution.local_diagnostics import iou

def extra_analysis(run):
    run=Path(run)
    with (run/'diagnosis.csv').open() as f:rows=list(csv.DictReader(f))
    traces={t['audio_filename']:t for t in map(json.loads,(run/'trace.jsonl').read_text().splitlines())}
    events=[e for t in traces.values() for e in t.get('events',[])]
    stages=Counter(e['stage'] for e in events)
    rejections=Counter(e['reason'] for e in events if e['stage']=='exp12_rejection')
    coverage=[e for e in events if e['stage']=='exp12_coverage']
    predicted_positive=sum(e['predicted_positive'] for e in coverage)
    resolved=sum(e['resolved'] for e in coverage)
    oracle=[];candidate_counts=[];gate_questions=0
    for row in rows:
        if row['label']!='1':continue
        trace=traces.get('conversation_'+row['conversation']+'.mp3',{})
        questions=trace.get('questions',[])
        idx=questions.index(row['question']) if row['question'] in questions else None
        candidates=[]
        for event in trace.get('events',[]):
            if event['stage']=='candidates' and event['index']==idx:candidates=event['candidates']
        gold=(float(row['gold_start']),float(row['gold_end']))
        oracle.append(max((iou(gold,(c['start'],c['end'])) for c in candidates),default=0))
        candidate_counts.append(len(candidates))
    all_conversations=len({r['conversation'] for r in rows})
    success=(len(coverage)==all_conversations and stages['exp12_failed']==0 and stages['exp12_missing']==0
             and predicted_positive>0 and resolved/predicted_positive>=.95)
    output=dict(execution_gate_passed=success,expected_conversations=all_conversations,
        stage_counts=dict(stages),rejection_reasons=dict(rejections),
        predicted_positive=predicted_positive,resolved_positive=resolved,
        grounding_coverage=resolved/predicted_positive if predicted_positive else 0,
        candidate_oracle_all_gold_positives=statistics.mean(oracle) if oracle else 0,
        candidate_coverage_all_gold_positives=sum(n>0 for n in candidate_counts),
        positive_denominator=len(oracle),
        note='Execution gate is not a quality gate. All gold positives count; no candidates means oracle zero.')
    (run/'exp12_execution.json').write_text(json.dumps(output,indent=2))
    return output
