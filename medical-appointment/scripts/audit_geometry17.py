"""CPU-only upper-bound audit of Exp17's calibrated word-pair representation."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from train_exp17 import load_data
from solution.reader17 import settings
from solution.span17 import geometry,overlaps

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--trace',type=Path,required=True);p.add_argument('--csv',type=Path,default=ROOT/'data/question_train.csv');a=p.parse_args()
    data=load_data(a.trace,a.csv);values=[]
    for d in data:
        s,e,valid=geometry(d['words'],dict(word_ids=list(range(len(d['words'])))),settings())
        for r in d['rows']:
            if r['label']=='1':values.append(float(np.where(valid,overlaps(s,e,(float(r['evidence_start']),float(r['evidence_end']))),0).max()))
    print(json.dumps(dict(positive_denominator=len(values),unwindowed_calibrated_word_grid_oracle=float(np.mean(values)),note='Gold-assisted representation ceiling; not predictions. Actual token-window oracle is reported during training.'),indent=2))
if __name__=='__main__':main()
