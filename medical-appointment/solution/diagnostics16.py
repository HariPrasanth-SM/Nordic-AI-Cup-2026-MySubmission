"""Offline before/after and full-denominator candidate diagnostics for Exp16."""
import csv,json,statistics
from collections import Counter
from pathlib import Path
from solution.local_diagnostics import iou,as_pair

def refinement_analysis(run):
    run=Path(run)
    with (run/'diagnosis.csv').open() as f:rows=list(csv.DictReader(f))
    traces={t['audio_filename']:t for t in map(json.loads,(run/'trace.jsonl').read_text().splitlines())}
    events=[e for t in traces.values() for e in t.get('events',[])];stages=Counter(e['stage'] for e in events)
    coverage=[e for e in events if e['stage']=='exp16_coverage']
    requested=sum(e['requested'] for e in coverage);proposed=sum(e['proposed'] for e in coverage);reviewed=sum(e['reviewed'] for e in coverage)
    details=[];class_changes=0
    for row in rows:
        t=traces.get('conversation_'+row['conversation']+'.mp3',{});qs=t.get('questions',[])
        idx=qs.index(row['question']) if row['question'] in qs else None
        bases=[e['results'] for e in t.get('events',[]) if e['stage']=='exp16_baseline']
        base=bases[-1][idx] if bases and idx is not None else None
        if base:class_changes+=int(int(base['p_yes']>.5)!=int(row['prediction']))
        if row['label']!='1':continue
        gold=(float(row['gold_start']),float(row['gold_end']));cs=[]
        for e in t.get('events',[]):
            if e['stage']=='exp16_candidates' and e['index']==idx:cs=e['candidates']
        old=iou(as_pair(base['span']),gold) if base else 0
        new=float(row['tiou']);oracle=max([iou((c['start'],c['end']),gold) for c in cs]+[old])
        calibrated=max([iou((round(c['start']+.1*(c['end']-c['start']),2),round(c['end']+.1,2)),gold) for c in cs]+[0])
        details.append(dict(question_id=row['question_id'],baseline_tiou=old,final_tiou=new,
            delta_tiou=new-old,candidate_oracle=oracle,calibrated_candidate_oracle=calibrated,candidates=len(cs),error_bucket=row['error_bucket']))
    expected=len({r['conversation'] for r in rows})
    success=(len(coverage)==expected and not stages['exp16_failed'] and not stages['exp16_skipped']
        and not stages['exp12_failed'] and not class_changes and requested>0 and reviewed==requested and proposed/requested>=.95)
    mean=lambda k:statistics.mean(d[k] for d in details) if details else 0
    result=dict(execution_gate_passed=success,expected_conversations=expected,stage_counts=dict(stages),
        requested=requested,proposed=proposed,reviewed=reviewed,classification_changes=class_changes,
        changed=sum(e['changed'] for e in coverage),baseline_tiou=mean('baseline_tiou'),final_tiou=mean('final_tiou'),
        candidate_oracle_all_gold_positives=mean('candidate_oracle'),calibrated_candidate_oracle_all_gold_positives=mean('calibrated_candidate_oracle'),positive_denominator=len(details),
        positive_spans_improved=sum(d['delta_tiou']>1e-6 for d in details),positive_spans_regressed=sum(d['delta_tiou']<-1e-6 for d in details),
        note='Execution gate only. All gold positives, including false negatives and missing spans, count. Candidates include baseline. Quality must improve separately.')
    (run/'exp16_execution.json').write_text(json.dumps(result,indent=2))
    with (run/'exp16_changes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(details[0]) if details else ['question_id']);writer.writeheader();writer.writerows(details)
    return result

def compare_saved_baseline(run):
    """Compare final API-rounded spans to the saved Exp15 response; labels offline only."""
    run=Path(run)
    with (run/'diagnosis.csv').open() as f:rows=list(csv.DictReader(f))
    traces={t['audio_filename']:t for t in map(json.loads,(run/'trace.jsonl').read_text().splitlines())}
    details=[]
    for row in rows:
        if row['label']!='1':continue
        t=traces.get('conversation_'+row['conversation']+'.mp3',{})
        previous=[e['response'] for e in t.get('events',[]) if e['stage']=='exp16_previous_response']
        if not previous:continue
        i=t['questions'].index(row['question']);p=previous[-1]
        span=None if p['evidence_start'][i] is None else (p['evidence_start'][i],p['evidence_end'][i])
        old=iou(span,(float(row['gold_start']),float(row['gold_end'])))
        new=float(row['tiou'])
        details.append(dict(question_id=row['question_id'],exp15_tiou=old,exp16_tiou=new,delta=new-old))
    if not details:return {}
    result=dict(positive_denominator=len(details),exp15_tiou=statistics.mean(d['exp15_tiou'] for d in details),
        exp16_tiou=statistics.mean(d['exp16_tiou'] for d in details),
        score_delta_with_frozen_classification=.6*statistics.mean(d['delta'] for d in details),
        old_zero_overlap=sum(d['exp15_tiou']==0 for d in details),
        zero_overlap_recovered=sum(d['exp15_tiou']==0 and d['exp16_tiou']>0 for d in details),
        old_strong_regressed_below_half=sum(d['exp15_tiou']>=.8 and d['exp16_tiou']<.5 for d in details))
    (run/'exp16_vs_exp15.json').write_text(json.dumps(result,indent=2))
    with (run/'exp16_vs_exp15.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
    return result
