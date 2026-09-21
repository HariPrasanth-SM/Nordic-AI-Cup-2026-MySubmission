"""Offline before/after and full-denominator candidate diagnostics for Exp13."""
import csv,json,statistics
from collections import Counter
from pathlib import Path
from solution.local_diagnostics import iou,as_pair

def refinement_analysis(run):
    run=Path(run)
    with (run/'diagnosis.csv').open() as f:rows=list(csv.DictReader(f))
    traces={t['audio_filename']:t for t in map(json.loads,(run/'trace.jsonl').read_text().splitlines())}
    events=[e for t in traces.values() for e in t.get('events',[])];stages=Counter(e['stage'] for e in events)
    coverage=[e for e in events if e['stage']=='exp13_coverage']
    requested=sum(e['requested'] for e in coverage);proposed=sum(e['proposed'] for e in coverage);reviewed=sum(e['reviewed'] for e in coverage)
    details=[];class_changes=0
    for row in rows:
        t=traces.get('conversation_'+row['conversation']+'.mp3',{});qs=t.get('questions',[])
        idx=qs.index(row['question']) if row['question'] in qs else None
        bases=[e['results'] for e in t.get('events',[]) if e['stage']=='exp13_baseline']
        base=bases[-1][idx] if bases and idx is not None else None
        if base:class_changes+=int(int(base['p_yes']>.5)!=int(row['prediction']))
        if row['label']!='1':continue
        gold=(float(row['gold_start']),float(row['gold_end']));cs=[]
        for e in t.get('events',[]):
            if e['stage']=='exp13_candidates' and e['index']==idx:cs=e['candidates']
        old=iou(as_pair(base['span']),gold) if base else 0
        new=float(row['tiou']);oracle=max([iou((c['start'],c['end']),gold) for c in cs]+[old])
        details.append(dict(question_id=row['question_id'],baseline_tiou=old,final_tiou=new,
            delta_tiou=new-old,candidate_oracle=oracle,candidates=len(cs),error_bucket=row['error_bucket']))
    expected=len({r['conversation'] for r in rows})
    success=(len(coverage)==expected and not stages['exp13_failed'] and not stages['exp13_skipped']
        and not stages['exp12_failed'] and not class_changes and requested>0 and reviewed==requested and proposed/requested>=.95)
    mean=lambda k:statistics.mean(d[k] for d in details) if details else 0
    result=dict(execution_gate_passed=success,expected_conversations=expected,stage_counts=dict(stages),
        requested=requested,proposed=proposed,reviewed=reviewed,classification_changes=class_changes,
        changed=sum(e['changed'] for e in coverage),baseline_tiou=mean('baseline_tiou'),final_tiou=mean('final_tiou'),
        candidate_oracle_all_gold_positives=mean('candidate_oracle'),positive_denominator=len(details),
        positive_spans_improved=sum(d['delta_tiou']>1e-6 for d in details),positive_spans_regressed=sum(d['delta_tiou']<-1e-6 for d in details),
        note='Execution gate only. All gold positives, including false negatives and missing spans, count. Candidates include baseline. Quality must improve separately.')
    (run/'exp13_execution.json').write_text(json.dumps(result,indent=2))
    with (run/'exp13_changes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(details[0]) if details else ['question_id']);writer.writeheader();writer.writerows(details)
    return result
