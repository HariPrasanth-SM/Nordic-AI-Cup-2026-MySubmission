import os,json,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from solution.types import Word,Segment
from solution.grounding12 import make_sources,resolve_quote,schema_for,AnchoredVerifier,LocalGrounder,tokens
from solution.evidence_experiment import transcript_hash

class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.segments=[Segment(0,3,'',[Word('Take',0,1),Word('10',1,2),Word('mg.',2,3)]),
                       Segment(8,12,'',[Word('Take',8,9),Word('10',9,10),Word('mg.',10,11),Word('Daily.',11,12)])]
        self.words,self.sources=make_sources(self.segments)
        self.config=dict(examples=1,source_forward_segments=2,max_prompt_chars=45000,repair_min_remaining_s=9,repair_max_tokens=900)
        self.bank=[dict(audio_sha256='other',transcript_hash='other',question='Dose?',context='Take 20 mg.',gold_quote='20 mg.')]
    def verifier(self,outputs):
        queue=iter(outputs);client=SimpleNamespace(config=self.config,call=lambda *args,**kwargs:next(queue))
        return AnchoredVerifier(client=client,bank=self.bank)
    def test_repeated_quote_respects_source(self):
        candidate,error=resolve_quote(dict(source='S001',quote='Take 10 mg.'),self.words,self.sources)
        self.assertIsNone(error);self.assertEqual(candidate['start'],8)
    def test_cross_segment_quote(self):
        candidate,error=resolve_quote(dict(source='S000',quote='mg. Take 10 mg.'),self.words,self.sources)
        self.assertIsNone(error);self.assertEqual((candidate['start'],candidate['end']),(2,11))
    def test_wrong_source_uniqueness_rule(self):
        c,error=resolve_quote(dict(source='S000',quote='Daily.'),self.words,self.sources)
        self.assertEqual(c['method'],'unique_global_recovery')
        # A third region with no occurrence: global repeated text must not jump to first.
        s=self.segments+[Segment(20,21,'',[Word('Hello.',20,21)])]
        w,src=make_sources(s)
        c,error=resolve_quote(dict(source='S002',quote='Take 10 mg.'),w,src)
        self.assertIsNone(c);self.assertEqual(error,'ambiguous_global')
    def test_decimal_and_placeholder(self):
        self.assertNotEqual(tokens('1.0 mg'),tokens('10 mg'))
        c,error=resolve_quote(dict(source='S000',quote='word'),self.words,self.sources)
        self.assertIsNone(c);self.assertEqual(error,'quote_not_found')
    def test_all_question_keys_required(self):
        s=schema_for(['q0','q1','q2'],self.sources)
        self.assertEqual(s['required'],['q0','q1','q2']);self.assertFalse(s['additionalProperties'])
        self.assertEqual(s['properties']['q0']['properties']['evidence']['items']['properties']['source']['enum'],['S000','S001'])
    def test_repair_missing_question_and_keep_valid(self):
        v=self.verifier([{'q0':{'supported':True,'evidence':[{'source':'S001','quote':'Take 10 mg.'}]}},
                         {'q1':{'supported':False,'evidence':[]}}])
        result=v.verify_batch(['Dose ten?','Dose twenty?'],self.segments,[],None,time.monotonic()+40)
        self.assertEqual([r.p_yes for r in result],[1,0]);self.assertEqual(result[0].span.start,8)
        self.assertEqual(sum(e['stage']=='exp12_repair_raw' for e in v.last_events),1)
    def test_unrepairable_quote_is_visible_not_silent_fallback(self):
        bad={'q0':{'supported':True,'evidence':[{'source':'S000','quote':'not in transcript'}]}}
        v=self.verifier([bad,bad]);r=v.verify_batch(['Dose?'],self.segments,[],None,time.monotonic()+40)
        self.assertIsNone(r[0].span)
        self.assertTrue(any(e['stage']=='exp12_missing_span' for e in v.last_events))
    def test_no_same_conversation_demonstration(self):
        v=self.verifier([]);v.audio_hash='same'
        v.bank=[dict(self.bank[0],audio_sha256='same'),dict(self.bank[0],transcript_hash=transcript_hash(self.words)),self.bank[0]]
        v.messages(['Dose?'],self.words,self.sources)
        e=v.last_events[-1];self.assertEqual(e['audio_hashes'],['other'])
    def test_remote_endpoint_rejected(self):
        with self.assertRaises(ValueError):LocalGrounder({'endpoint':'https://example.com/v1/chat/completions'})
    def test_local_http_contract(self):
        from http.server import BaseHTTPRequestHandler,HTTPServer
        import threading
        captured=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                captured.append(body)
                content=json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'{"ready":true}'}}],
                                    'model':'medical-grounder','usage':{'completion_tokens':5}}).encode()
                self.send_response(200);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content)
            def log_message(self,*args):pass
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            config=dict(endpoint=f'http://127.0.0.1:{server.server_port}/v1/chat/completions',model='medical-grounder',
                        temperature=0,seed=1234,max_tokens=100,call_timeout_s=5)
            client=LocalGrounder(config);client.warmup()
            self.assertFalse(captured[0]['chat_template_kwargs']['enable_thinking'])
            self.assertEqual(captured[0]['response_format']['schema']['required'],['ready'])
        finally:server.shutdown();server.server_close();thread.join()

    def test_full_denominator_candidate_oracle_and_execution_gate(self):
        from solution.diagnostics12 import extra_analysis
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'diagnosis.csv').write_text('question_id,conversation,question,label,gold_start,gold_end\n'
                                             'a,sample_1,Q1?,1,1,2\nb,sample_1,Q2?,1,4,5\n')
            trace={'audio_filename':'conversation_sample_1.mp3','questions':['Q1?','Q2?'],'events':[
                {'stage':'candidates','index':0,'candidates':[{'start':1,'end':2}]},
                {'stage':'exp12_coverage','predicted_positive':2,'resolved':1,'total':2}]}
            (root/'trace.jsonl').write_text(json.dumps(trace)+'\n')
            result=extra_analysis(root)
            self.assertEqual(result['candidate_oracle_all_gold_positives'],.5)
            self.assertFalse(result['execution_gate_passed'])

    def test_exp12_does_not_load_old_llm(self):
        with patch.dict(os.environ,{'MEDICAL_APPT_SKIP_MODEL_LOAD':'1','MEDICAL_APPT_EXPERIMENT':'exp12'}):
            from solution.config import load_config
            import solution.pipeline as pipeline
            config=load_config('workstation_32gb','20260918T175818Z_49f0c52')
            client=SimpleNamespace(warmup=lambda:None)
            instance=SimpleNamespace(client=client)
            with patch('solution.grounding12.AnchoredVerifier',return_value=instance) as ctor:
                self.assertIs(pipeline._build_verifier(config,SimpleNamespace()),instance)
            self.assertEqual(config.threshold.fixed_value,.5)

if __name__=='__main__':unittest.main()
