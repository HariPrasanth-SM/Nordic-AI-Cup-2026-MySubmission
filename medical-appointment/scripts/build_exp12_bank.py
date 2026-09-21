"""Offline teacher snippets with measured word-span fit and consultation exclusion."""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.grounding12 import make_sources
from solution.evidence_experiment import transcript_hash
from solution.types import Word,Segment
from solution.local_diagnostics import iou

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--trace',required=True,type=Path)
    p.add_argument('--csv',default='data/question_train.csv',type=Path)
    p.add_argument('--output',default='models/annotation_bank_exp12.json',type=Path)
    a=p.parse_args()
    traces={r['audio_filename']:r for r in map(json.loads,a.trace.read_text().splitlines())}
    with a.csv.open() as f:rows=list(csv.DictReader(f))
    examples=[];excluded=[]
    for row in rows:
        if row['label']!='1':continue
        t=traces.get('conversation_'+row['transcript_id']+'.mp3')
        if t is None:continue
        segments=[Segment(s['start'],s['end'],s['text'],[Word(**w) for w in s['words']]) for s in t['segments']]
        words,sources=make_sources(segments);gold=(float(row['evidence_start']),float(row['evidence_end']))
        selected=[i for i,w in enumerate(words) if gold[0]<=(w.start+w.end)/2<=gold[1]]
        if not selected:excluded.append((row['question_id'],'no_word'));continue
        lo,hi=selected[0],selected[-1]
        fit=iou(gold,(words[lo].start,words[hi].end))
        if gold[0]==0 and gold[1]<.5:
            excluded.append((row['question_id'],'suspicious_initial_subsecond_label'));continue
        if fit<.75:excluded.append((row['question_id'],'word_boundary_fit_below_075'));continue
        examples.append(dict(audio_sha256=t['audio_sha256'],transcript_hash=transcript_hash(words),
            question=row['question'],gold_quote=' '.join(w.text for w in words[lo:hi+1]),
            context=' '.join(w.text for w in words[max(0,lo-25):min(len(words),hi+26)]),
            question_id=row['question_id'],word_boundary_fit=fit))
    if not examples:raise SystemExit('No training examples could be built')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(examples,indent=2))
    a.output.with_suffix('.audit.json').write_text(json.dumps(dict(examples=len(examples),excluded=excluded),indent=2))
    print(f'Wrote {len(examples)} examples; excluded {len(excluded)} low-fit/suspicious spans.')
    print('Excluded labels remain scored in evaluation. This bank contains only development labels.')
if __name__=='__main__':main()
