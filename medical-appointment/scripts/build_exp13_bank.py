"""Offline contrast bank builder. Gold labels are never read by the runtime."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from solution.local_diagnostics import iou,text_at

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--trace',required=True,type=Path)
    p.add_argument('--csv',type=Path,default=ROOT/'data/question_train.csv')
    p.add_argument('--bank',type=Path,default=ROOT/'models/annotation_bank_exp12.json')
    p.add_argument('--output',type=Path,default=ROOT/'models/contrast_bank_exp13.json');a=p.parse_args()
    traces={t['audio_sha256']:t for t in map(json.loads,a.trace.read_text().splitlines())}
    with a.csv.open() as f:rows={r['question_id']:r for r in csv.DictReader(f)}
    bank=json.loads(a.bank.read_text())
    for d in bank:
        t=traces.get(d['audio_sha256']);r=rows.get(d['question_id']);d['contrast']={}
        if not t or not r:continue
        idx=t['questions'].index(d['question']);s=t['response']['evidence_start'][idx];e=t['response']['evidence_end'][idx]
        if s is None:continue
        fit=iou((s,e),(float(r['evidence_start']),float(r['evidence_end'])))
        if fit>=.75:continue
        words=[w for seg in t['segments'] for w in seg['words']]
        d['contrast']={'lower_overlap_quote':text_at(words,(s,e)),
          'interpretation':'Different occurrence; may still be semantically true.' if fit==0 else 'Boundary mismatch; learn annotated extent, not the longer context.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(bank,indent=2))
    print(len(bank),'examples;',sum(bool(d['contrast']) for d in bank),'contrasts')
if __name__=='__main__':main()
