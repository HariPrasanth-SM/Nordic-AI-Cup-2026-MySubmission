"""No GPU required. Run: python -m unittest discover -s tests -p test_evidence_experiments.py"""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from solution.types import Word,Segment,Span,VerifierResult
from solution.evidence_experiment import EvidenceExperiment,choose_examples,transcript_hash
from solution.forced_timing import map_alignment
from solution.local_diagnostics import iou,word_oracle,analyze
from solution import experiment_trace

class EvidenceTests(unittest.TestCase):
    def test_exclude_current_conversation(self):
        words=[Word('Yes',0,1)]
        bank=[dict(audio_sha256='same',transcript_hash='other',question='dose'),
              dict(audio_sha256='other',transcript_hash=transcript_hash(words),question='dose'),
              dict(audio_sha256='safe',transcript_hash='safe',question='dose')]
        self.assertEqual([x['audio_sha256'] for x in choose_examples(bank,['dose'],words,'same')],['safe'])

    def test_alignment_split_and_repeated_words(self):
        words=[Word('twenty-five',1,2),Word('yes',3,4),Word('yes',7,8)]
        aligned=[SimpleNamespace(text=t,start_time=s,end_time=e) for t,s,e in
                 [('twenty',1.1,1.4),('five',1.4,2.1),('yes',3.1,4.1),('yes',7.1,8.1)]]
        result=map_alignment(words,aligned)
        self.assertEqual((result[0].start,result[0].end),(1.1,2.1))
        self.assertEqual(result[2].start,7.1)
        aligned[0].text='thirty'
        with self.assertRaises(ValueError):map_alignment(words,aligned)

    def test_full_oracle_and_missing(self):
        words=[dict(text='one',start=1,end=2),dict(text='two',start=3,end=5)]
        self.assertEqual(word_oracle(words,(1,5))[0],1)
        self.assertEqual(iou((1,5),None),0)
        self.assertEqual(iou((1,2),(3,4)),0)

    def test_selector_preserves_classification_and_rejects_bad_ids(self):
        words=[Word('Take',0,1),Word('five',1,2),Word('mg.',2,3)]
        base=[VerifierResult(.99,Span(0,1),.6,'Take')]
        for endpoint,expected in [('mg.',3),('wrong',1)]:
            model=SimpleNamespace(generate_json=lambda _:dict(items=[dict(index=0,spans=[
                dict(start=1,end=2,first='five',last=endpoint)])]))
            verifier=SimpleNamespace(verify_batch=lambda *args:base,llm_model=model)
            with tempfile.TemporaryDirectory() as folder:
                bank=Path(folder)/'bank.json';bank.write_text(json.dumps([dict(audio_sha256='other',
                    transcript_hash='other',question='dose',context='Take ten mg.',gold_quote='ten mg.')]))
                with patch.dict(os.environ,{'MEDICAL_ANNOTATION_BANK':str(bank)}):
                    selector=EvidenceExperiment(verifier)
                out=selector.verify_batch(['Is dose five mg?'],[Segment(0,3,'Take five mg.',words)],[],None,time.monotonic()+50)
                self.assertEqual(out[0].p_yes,.99);self.assertEqual(out[0].span.end,expected)

    def test_pipeline_traces_real_response_without_loading_models(self):
        import base64
        with patch.dict(os.environ,{'MEDICAL_APPT_SKIP_MODEL_LOAD':'1','MEDICAL_APPT_EXPERIMENT':'baseline'}):
            import solution.pipeline as pipeline
        from dtos import ASRQuestionRequestDto
        segments=[Segment(0,3,'Take five mg.',[Word('Take',0,1),Word('five',1,2),Word('mg.',2,3)])]
        asr=SimpleNamespace(transcribe_bytes=lambda _:segments)
        retrieval=SimpleNamespace(embed_passages=lambda _:[])
        verifier=SimpleNamespace(verify_batch=lambda *args:[VerifierResult(.99,Span(1,3),.6,'five mg.')])
        with tempfile.TemporaryDirectory() as folder:
            trace=Path(folder)/'trace.jsonl'
            with patch.dict(os.environ,{'MEDICAL_LOCAL_TRACE':str(trace)}), \
                 patch.object(pipeline,'ASR_MODEL',asr),patch.object(pipeline,'RETRIEVER',retrieval), \
                 patch.object(pipeline,'VERIFIER',verifier),patch.object(pipeline,'EXPERIMENT','baseline'), \
                 patch.object(pipeline,'ALIGNER',None):
                response=pipeline._predict_impl(ASRQuestionRequestDto(audio_base64=base64.b64encode(b'fake').decode(),
                    audio_filename='sample.mp3',questions=['Five mg?']))
            self.assertEqual(response.answers,[True])
            self.assertEqual(response.evidence_start,[1])
            saved=json.loads(trace.read_text())
            self.assertEqual(saved['response']['evidence_end'],[3])
            self.assertEqual(len(saved['segments'][0]['words']),3)

    def test_experiment_config_applies_after_calibration(self):
        from solution.config import load_config
        with patch.dict(os.environ,{'MEDICAL_APPT_EXPERIMENT':'exp10'}):
            config=load_config('workstation_32gb','20260918T175818Z_49f0c52')
        self.assertEqual(config.llm.n_ctx,16384)
        self.assertEqual(config.threshold.fixed_value,.06)

    def test_trace_disabled(self):
        with patch.dict(os.environ,{},clear=True):
            experiment_trace.reset();experiment_trace.event('sensitive',text='hello')
            self.assertEqual(experiment_trace._EVENTS,[])

    def test_diagnosis_scores_unsent_and_false_negative(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=root/'gold.csv'
            data.write_text('question_id,transcript_id,question,answer,label,question_type,evidence_start,evidence_end\n'
                'a,sample_1,Q?,yes,1,positive,1,2\nb,sample_1,N?,no,0,hard_negative,,\n'
                'c,sample_2,Q2?,yes,1,positive,1,2\n')
            r=dict(audio_filename='conversation_sample_1.mp3',questions=['Q?','N?'],question_ids=['a','b'],
                   predictions=[0,0],spans=[None,None],latency_ms=5,error=None)
            (root/'predictions.jsonl').write_text(json.dumps(r)+'\n')
            result=analyze(root,data)
            self.assertEqual(result['failures'],1)
            self.assertEqual(result['accuracy'],1/3)
            self.assertEqual(result['mean_tiou'],0)
            self.assertAlmostEqual(result['score'],.4/3)

if __name__=='__main__':unittest.main()
