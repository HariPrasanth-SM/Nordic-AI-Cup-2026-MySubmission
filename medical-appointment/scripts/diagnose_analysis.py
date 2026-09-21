"""Rebuild diagnostics without rerunning models; optional paired conversation bootstrap."""
import argparse,csv,json,random,statistics,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.local_diagnostics import analyze

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--csv',default='data/question_train.csv')
    p.add_argument('--compare',type=Path,help='baseline run directory')
    a=p.parse_args();print(json.dumps(analyze(a.run,a.csv),indent=2))
    if a.compare:
        analyze(a.compare,a.csv)
        new={r['question_id']:r for r in csv.DictReader((a.run/'diagnosis.csv').open())}
        old={r['question_id']:r for r in csv.DictReader((a.compare/'diagnosis.csv').open())}
        if new.keys()!=old.keys():raise SystemExit('Question sets differ')
        groups=defaultdict(list)
        for key,row in new.items():groups[row['conversation']].append(key)
        names=list(groups)
        def score(rows):
            return .4*statistics.mean(float(r['correct']) for r in rows)+.6*statistics.mean(float(r['tiou']) for r in rows if r['label']=='1')
        rng=random.Random(1729);deltas=[]
        for _ in range(2000):
            keys=[key for c in rng.choices(names,k=len(names)) for key in groups[c]]
            deltas.append(score([new[k] for k in keys])-score([old[k] for k in keys]))
        deltas.sort()
        result=dict(delta_score=score(list(new.values()))-score(list(old.values())),
            paired_conversation_bootstrap_95_percent=[deltas[50],deltas[1949]],
            note='Development uncertainty only; does not correct repeated tuning or shared demonstration-bank dependence.')
        (a.run/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
