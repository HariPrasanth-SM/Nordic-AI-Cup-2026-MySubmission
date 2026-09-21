"""Local reader coverage check, separate from quality and live timing."""
import json
from collections import Counter
from pathlib import Path

def reader_analysis(run):
    run=Path(run);ts=[json.loads(x) for x in (run/'trace.jsonl').read_text().splitlines()]
    counts=Counter(e['stage'] for t in ts for e in t.get('events',[]))
    coverage=[e for t in ts for e in t.get('events',[]) if e['stage']=='exp17_coverage']
    summary=json.loads((run/'diagnosis.json').read_text())
    result=dict(execution_gate_passed=len(ts)==summary.get('response_conversations',0) and summary.get('failures',1)==0
        and len(coverage)==len(ts) and all(x['reviewed']==x['requested'] for x in coverage) and not any(counts[x] for x in ('exp17_failed','exp17_skipped','exp12_failed','exp13_failed','exp13_skipped')),
        conversations=len(ts),stage_counts=dict(counts),requested=sum(x['requested'] for x in coverage),
        reviewed=sum(x['reviewed'] for x in coverage),changed=sum(x['changed'] for x in coverage),note='Execution gate only; use held-out reader scores for quality.')
    (run/'exp17_execution.json').write_text(json.dumps(result,indent=2));return result
