"""Supervised reader competes with the actual calibrated Exp15 result."""
import json,os,time
from pathlib import Path
from dataclasses import asdict
from solution.types import Span,VerifierResult,Word
from solution.experiment_trace import event
from solution.span17 import windows,decode
ROOT=Path(__file__).resolve().parents[1]
def settings():
    return json.loads(Path(os.environ.get('MEDICAL_EXP17_CONFIG',str(ROOT/'configs/exp17_reader.json'))).read_text())

class Reader:
    def __init__(self,path=None,cfg=None,model=None,tokenizer=None):
        import torch
        from transformers import AutoTokenizer,AutoModelForQuestionAnswering
        self.cfg=cfg or settings();path=Path(path or self.cfg['model_path'])
        if not path.is_absolute():path=ROOT/path
        if model is None:
            trained=path/'exp17_metadata.json'
            if not trained.exists():raise FileNotFoundError('Train the Exp17 reader first: '+str(trained))
            self.metadata=json.loads(trained.read_text())
            if self.metadata.get('kind')!='final':raise ValueError('Deployment requires the final model, not a fold checkpoint')
            for k in ('max_length','stride','max_words','max_duration_s','start_fraction','end_offset_s'):
                if self.metadata['config'][k]!=self.cfg[k]:raise ValueError('Training/inference config differs: '+k)
            tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True)
            model=AutoModelForQuestionAnswering.from_pretrained(path,local_files_only=True)
        self.tokenizer=tokenizer;self.model=model.to(self.cfg['device']).eval()
        if self.cfg['device'].startswith('cuda'):
            self.model.to(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        self.torch=torch
    def infer(self,questions,words,baselines,deadline):
        torch=self.torch;features=[windows(self.tokenizer,q,words,self.cfg) for q in questions]
        flat=[f for fs in features for f in fs];all_logits=[];batch=self.cfg['batch_size']
        for lo in range(0,len(flat),batch):
            if time.monotonic()>=deadline-1:raise TimeoutError('Exp17 reader budget exhausted')
            chunk=flat[lo:lo+batch]
            inputs={k:torch.tensor([f['inputs'][k] for f in chunk],device=self.cfg['device']) for k in chunk[0]['inputs']}
            with torch.inference_mode():pred=self.model(**inputs)
            all_logits.extend(zip(pred.start_logits.float().cpu().numpy(),pred.end_logits.float().cpu().numpy()))
        output=[];offset=0
        for fs,base in zip(features,baselines):
            output.append(decode(words,fs,all_logits[offset:offset+len(fs)],self.cfg,base));offset+=len(fs)
        return output

class ReaderVerifier:
    def __init__(self,base,reader=None):
        self.base=base;self.reader=reader or Reader();self.audio_hash=''
        if reader is None:
            self.reader.infer(['Does medication continue?'],[Word('Medication',0,1),Word('continues.',1,2)],[None],time.monotonic()+30)
    def verify_batch(self,questions,segments,windows_,vectors,deadline):
        self.base.audio_hash=self.audio_hash
        baseline=self.base.verify_batch(questions,segments,windows_,vectors,deadline)
        return self.refine(questions,segments,baseline,deadline)
    def refine(self,questions,segments,baseline,deadline):
        event('exp17_baseline',results=[asdict(r) for r in baseline])
        if deadline-time.monotonic()<self.reader.cfg['min_remaining_s']:
            event('exp17_skipped',reason='budget');return baseline
        wanted=[i for i,r in enumerate(baseline) if r.p_yes>.5]
        if not wanted:event('exp17_coverage',requested=0,reviewed=0,changed=0);return baseline
        words=[w for s in segments for w in s.words]
        try:
            records=self.reader.infer([questions[i] for i in wanted],words,
                [None if baseline[i].span is None else (baseline[i].span.start,baseline[i].span.end) for i in wanted],deadline)
            output=list(baseline);changed=0
            for i,record in zip(wanted,records):
                event('exp17_review',index=i,**record)
                event('candidates',index=i,candidates=record['candidates'])
                if record['accepted']:
                    c=record['candidate'];r=baseline[i];output[i]=VerifierResult(r.p_yes,Span(c['start'],c['end']),r.expected_tiou,c['text'])
                    changed+=int(output[i].span!=r.span)
            event('exp17_coverage',requested=len(wanted),reviewed=len(records),changed=changed)
            return output
        except Exception as exc:
            event('exp17_failed',reason=str(exc));return baseline
