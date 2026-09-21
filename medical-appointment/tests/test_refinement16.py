import json,time,unittest
from types import SimpleNamespace
from solution.refinement16 import ContrastiveVerifier,accepted
from test_refinement13 import CFG,SEGS,BASE,DEMO
class Client:
    config={}
    def __init__(self,unknown=False):self.calls=[];self.unknown=unknown
    def call(self,messages,schema,deadline,max_tokens=None):
        self.calls.append(messages)
        if len(self.calls)==1:
            q=dict(source='S000',quote='Take 10 mg daily.')
            return {'q0':dict(core=q,context=q,alternative=None,alternative_context=None,alternative2=None)}
        text=messages[-1]['content']
        assert 'CURRENT TRANSCRIPT:' not in text and 'Existing quote:' not in text
        cs=json.loads(text.split('CANDIDATES:\n')[1])['q0'];c=next(c for c in cs if c['text']=='Take 10 mg daily.')
        return {'q0':dict(candidate=c['id'],covers_all_details=True,judgments={x['id']:'unknown' if self.unknown else 'entailed' for x in cs},critical_fact='10 mg',replacement_fact='20 mg',counterfactual_relation='contradicted')}
class Tests(unittest.TestCase):
    def test_selection_preserves_classes_and_removes_context(self):
        client=Client();v=ContrastiveVerifier(base=SimpleNamespace(client=client),client=client,bank=[DEMO],config=CFG)
        out=v.refine(['Take 10 mg daily?','Take 20 mg?'],SEGS,BASE,time.monotonic()+60)
        self.assertEqual([x.p_yes for x in out],[1,0]);self.assertEqual(out[0].span.start,1);self.assertEqual(out[0].span.end,5)
    def test_unknown_cannot_pass(self):
        client=Client(True);v=ContrastiveVerifier(base=SimpleNamespace(client=client),client=client,bank=[DEMO],config=CFG)
        self.assertEqual(v.refine(['Q','N'],SEGS,BASE,time.monotonic()+60),BASE)
    def test_invented_critical_fact_rejected(self):
        self.assertFalse(accepted(dict(covers_all_details=True,judgments={'C0':'entailed'},critical_fact='20 mg',counterfactual_relation='contradicted'),dict(id='C0',text='10 mg')))
if __name__=='__main__':unittest.main()
