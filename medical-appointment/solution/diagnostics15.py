"""Full-run calibration execution checks, independent of model quality."""
import csv,json
from collections import Counter
from pathlib import Path
from solution.local_diagnostics import iou,as_pair

def calibration_analysis(run):
    p=Path(run);ts=list(map(json.loads,(p/'trace.jsonl').read_text().splitlines()))
    with (p/'diagnosis.csv').open() as f:rows=list(csv.DictReader(f))
    es=[e for t in ts for e in t.get('events',[])];counts=Counter(e['stage'] for e in es)
    expected=len({r['conversation'] for r in rows});cal=[e for e in es if e['stage']=='exp15_calibration']
    change=sum(e['classification_changes'] for e in cal)
    failed=[e for e in es if e['stage'] in {'exp12_failed','exp12_missing','exp13_failed','exp13_skipped'}]
    cfgs=[e['config'] for e in cal]
    byname={t['audio_filename']:t for t in ts};deltas=[];base_tiou=[];final_tiou=[]
    for row in rows:
        if row['label']!='1':continue
        t=byname.get('conversation_'+row['conversation']+'.mp3',{});qs=t.get('questions',[])
        idx=qs.index(row['question']) if row['question'] in qs else None
        baselines=[e['results'] for e in t.get('events',[]) if e['stage']=='exp15_baseline']
        old=as_pair(baselines[-1][idx]['span']) if baselines and idx is not None else None
        gold=(float(row['gold_start']),float(row['gold_end']));before=iou(tuple(round(x,2) for x in old) if old else None,gold);after=float(row['tiou'])
        base_tiou.append(before);final_tiou.append(after);deltas.append(after-before)
    # Cached replay verifies its saved base; live run includes all base events.
    coverage=[e for e in es if e['stage']=='exp13_coverage']
    coverage_ok=all(e['reviewed']==e['requested'] and (not e['requested'] or e['proposed']/e['requested']>=.95) for e in coverage)
    result=dict(execution_gate_passed=len(cal)==expected and not change and not failed and coverage_ok,
        expected_conversations=expected,calibrated_conversations=len(cal),classification_changes=change,
        baseline_tiou=sum(base_tiou)/len(base_tiou),final_tiou=sum(final_tiou)/len(final_tiou),
        improved=sum(x>1e-6 for x in deltas),regressed=sum(x<-1e-6 for x in deltas),
        stage_counts=dict(counts),changed_spans=sum(e['changed'] for e in cal),
        configuration=cfgs[0] if cfgs else None,
        note='Calibration execution only, not proof of quality. Cached inherited timings do not measure live performance.')
    (p/'exp15_execution.json').write_text(json.dumps(result,indent=2));return result
