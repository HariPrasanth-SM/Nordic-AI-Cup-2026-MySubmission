"""Exercise actual model schema, copied quotes and source disambiguation on a toy case."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.grounding12 import LocalGrounder,make_sources,schema_for,resolve_quote
from solution.types import Segment,Word

def main():
    client=LocalGrounder();client.warmup()
    words1=[Word('The',0,1),Word('dose',1,2),Word('is',2,3),Word('10',3,4),Word('mg.',4,5)]
    words2=[Word('Continue',10,11),Word('10',11,12),Word('mg',12,13),Word('daily.',13,14)]
    words,sources=make_sources([Segment(0,5,'',words1),Segment(10,14,'',words2)])
    messages=[{'role':'user','content':
        'Transcript: [S000] The dose is 10 mg. [S001] Continue 10 mg daily.\n'
        'q0: Is the dose 10 mg? Answer supported=true, copying the complete first sentence from S000.\n'
        'q1: Is the dose 20 mg? Answer supported=false and evidence=[]. Return both keys using the schema.'}]
    raw=client.call(messages,schema_for(['q0','q1'],sources),time.monotonic()+30,max_tokens=250)
    assert raw['q0']['supported'] is True and raw['q1']['supported'] is False,raw
    assert raw['q1']['evidence']==[],raw
    assert raw['q0']['evidence'],raw
    match,error=resolve_quote(raw['q0']['evidence'][0],words,sources)
    assert error is None and match['start']==0 and match['end']==5,(raw,error,match)
    print(json.dumps(raw,indent=2));print('PASS: local model responds, schema works, full quote maps to the correct occurrence.')
if __name__=='__main__':main()
