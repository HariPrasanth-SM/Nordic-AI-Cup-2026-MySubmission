"""Build annotation demonstrations from a traced baseline run, offline only."""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.types import Word
from solution.evidence_experiment import transcript_hash


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--trace',required=True,type=Path)
    p.add_argument('--csv',default='data/question_train.csv',type=Path)
    p.add_argument('--output',default='models/annotation_bank.json',type=Path)
    a=p.parse_args()
    gold=list(csv.DictReader(a.csv.open()))
    traces={r['audio_filename']:r for r in map(json.loads,a.trace.read_text().splitlines())}
    examples=[]
    for row in gold:
        if row['label']!='1': continue
        name='conversation_'+row['transcript_id']+'.mp3'
        t=traces.get(name)
        if t is None: continue
        words=[Word(**w) for s in t['segments'] for w in s['words']]
        start,end=float(row['evidence_start']),float(row['evidence_end'])
        # Midpoint membership makes the teacher span independent of a duration prior.
        chosen=[i for i,w in enumerate(words) if start <= (w.start+w.end)/2 <= end]
        if not chosen: continue
        lo,hi=chosen[0],chosen[-1]
        context=' '.join(w.text for w in words[max(0,lo-22):min(len(words),hi+23)])
        examples.append(dict(audio_sha256=t['audio_sha256'],transcript_hash=transcript_hash(words),
            question=row['question'],context=context,
            gold_quote=' '.join(w.text for w in words[lo:hi+1]),
            gold_duration=end-start,question_id=row['question_id']))
    if not examples: raise SystemExit('No matching gold/trace examples')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(examples,indent=2,ensure_ascii=False))
    print(f'{len(examples)} demonstrations from {len(set(e["audio_sha256"] for e in examples))} conversations -> {a.output}')
    print('Same-audio and same-transcript examples are excluded at inference.')
    print('This is development-set leave-conversation-out prompting, not an independent holdout.')

if __name__=='__main__':main()
