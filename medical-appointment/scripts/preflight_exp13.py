"""Exercise the actual server's Exp13 schema, extraction, and selection transport."""
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.refinement13 import ContrastiveVerifier
from solution.types import Word,Segment,Span,VerifierResult
v=ContrastiveVerifier();v.client.warmup()
words=[Word(t,i,i+1) for i,t in enumerate(['Hello.','Take','10','mg','daily.','Goodbye.'])]
segments=[Segment(0,6,' '.join(w.text for w in words),words)]
base=[VerifierResult(1,Span(0,6),.6,'Hello. Take 10 mg daily. Goodbye.')]
result=v.refine(['Should the patient take 10 mg daily?'],segments,base,time.monotonic()+45)
coverage=[e for e in v.last_events if e['stage']=='exp13_coverage']
if not coverage or coverage[-1]['proposed']!=1 or coverage[-1]['reviewed']!=1:
    print(v.last_events);raise SystemExit('FAIL: Exp13 did not complete extraction and selection')
assert result[0].p_yes==1 and result[0].span is not None
print('PASS: Exp13 schema/transport/selection completed. This does not measure task quality.')
print(result[0])
