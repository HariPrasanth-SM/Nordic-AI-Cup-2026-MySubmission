"""Train/evaluate a timestamp-supervised QA reader with consultation-group folds.
No LLM calls or audio decoding. Never score a model on its own training consultations.
"""
import argparse,copy,csv,gc,hashlib,json,math,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from solution.types import Word
from solution.span17 import windows,supervision,geometry,overlaps
from solution.local_diagnostics import analyze,iou

def load_data(trace,csv_path):
    traces=[json.loads(l) for l in trace.read_text().splitlines() if l.strip()]
    groups={}
    for row in csv.DictReader(csv_path.open()):groups.setdefault('conversation_'+row['transcript_id']+'.mp3',[]).append(row)
    seen=set();data=[]
    for t in traces:
        name=t['audio_filename']
        if name in seen:raise ValueError('Duplicate conversation: '+name)
        seen.add(name);rows=groups[name]
        if [r['question'] for r in rows]!=t['questions']:raise ValueError('Question order mismatch: '+name)
        stages={e['stage'] for e in t.get('events',[])}
        if 'exp15_calibration' not in stages or stages & {'exp16_config','exp17_baseline'}:
            raise ValueError('Use the best Exp15 trace, not Exp16 or a previously modified trace')
        words=[Word(**w) for s in t['segments'] for w in s['words']]
        data.append(dict(trace=t,rows=rows,words=words))
    if seen!=set(groups):raise ValueError('Trace must cover exactly the gold CSV conversations')
    # Union identical audio OR normalized transcripts to keep duplicates together.
    parents=list(range(len(data)))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    aliases={}
    for i,d in enumerate(data):
        text=' '.join(w.text.strip().lower() for w in d['words'])
        keys=['text:'+hashlib.sha256(text.encode()).hexdigest()]
        if d['trace'].get('audio_sha256'):keys.append('audio:'+d['trace']['audio_sha256'])
        for key in keys:
            if key in aliases:parents[find(i)]=find(aliases[key])
            aliases[key]=i
    for i,d in enumerate(data):d['group']=find(i)
    return data

def train_model(items,tokenizer,args,cfg,fold):
    import torch
    from transformers import AutoModelForQuestionAnswering
    torch.manual_seed(args.seed+fold);random.seed(args.seed+fold)
    model=AutoModelForQuestionAnswering.from_pretrained(args.model,revision=args.revision).to(args.device)
    # Preserve lexical representations; adapt the upper half and QA head.
    for name,p in model.named_parameters():
        if '.embeddings.' in name:p.requires_grad=False
        if '.encoder.layer.' in name:
            layer=int(name.split('.encoder.layer.')[1].split('.')[0])
            if layer<6:p.requires_grad=False
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=.01)
    rng=random.Random(args.seed+fold);model.train();step=0
    total=args.epochs*math.ceil(len(items)/args.batch_size)
    use_amp=args.device.startswith('cuda') and torch.cuda.is_bf16_supported()
    for epoch in range(args.epochs):
        order=list(range(len(items)));rng.shuffle(order);losses=[]
        for lo in range(0,len(order),args.batch_size):
            batch=[items[i] for i in order[lo:lo+args.batch_size]]
            inputs={k:torch.tensor([x['feature']['inputs'][k] for x in batch],device=args.device) for k in batch[0]['feature']['inputs']}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda' if args.device.startswith('cuda') else 'cpu',dtype=torch.bfloat16,enabled=use_amp):
                output=model(**inputs)
            terms=[]
            for j,x in enumerate(batch):
                f=x['feature'];target=x['target'];sl=output.start_logits[j].float();el=output.end_logits[j].float()
                if target['null']:
                    loss=-.5*(torch.log_softmax(sl,0)[f['cls']]+torch.log_softmax(el,0)[f['cls']])
                else:
                    sp=torch.tensor(target['start'],device=args.device,dtype=torch.float32)
                    ep=torch.tensor(target['end'],device=args.device,dtype=torch.float32)
                    loss=-.5*((sp*torch.log_softmax(sl,0)[f['start_tokens']]).sum()+(ep*torch.log_softmax(el,0)[f['end_tokens']]).sum())
                terms.append(loss*target['weight'])
            loss=torch.stack(terms).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            step+=1;scale=min(1.,step/max(1,.1*total))*max(0.,(total-step+1)/max(1,.9*total))
            for group in optimizer.param_groups:group['lr']=args.lr*scale
            optimizer.step();losses.append(float(loss.detach()))
        print(json.dumps(dict(fold=fold,epoch=epoch+1,loss=float(np.mean(losses)),windows=len(items))),flush=True)
    del optimizer
    return model.eval()

def evaluate_fold(model,tokenizer,heldout,cfg,args,out,fold):
    from solution.reader17 import Reader
    reader=Reader(cfg=cfg,model=model,tokenizer=tokenizer);records=[]
    for d in heldout:
        t=d['trace'];response=copy.deepcopy(t['response']);base=copy.deepcopy(response)
        wanted=[i for i,y in enumerate(response['answers']) if y]
        starts=response['evidence_start'];ends=response['evidence_end']
        begin=time.monotonic()
        result=reader.infer([t['questions'][i] for i in wanted],d['words'],
            [None if starts[i] is None else (starts[i],ends[i]) for i in wanted],begin+600)
        elapsed=time.monotonic()-begin;events=[];changed=0
        for i,r in zip(wanted,result):
            events.append(dict(stage='exp17_review',index=i,**r));events.append(dict(stage='candidates',index=i,candidates=r['candidates']))
            if r['accepted']:
                c=r['candidate'];changed+=int((starts[i],ends[i])!=(c['start'],c['end']))
                starts[i],ends[i]=c['start'],c['end']
        events.append(dict(stage='exp17_coverage',requested=len(wanted),reviewed=len(result),changed=changed))
        rec=copy.deepcopy(t);rec.update(response=response,elapsed_s=None,events=events,exp17_fold=fold)
        # Avoid inherited stale Exp15 result spans and inherited timing claims.
        for i,r in enumerate(rec['results']):
            r['span']=None if starts[i] is None else dict(start=starts[i],end=ends[i])
            selected=next((x for j,x in zip(wanted,result) if j==i and x['accepted']),None)
            if selected:r['quote']=selected['candidate']['text']
        rec['events'].append(dict(stage='exp17_saved_exp15',response=base,reader_elapsed_s=elapsed))
        records.append(rec)
        print(t['audio_filename'],'reader-only seconds',round(elapsed,3),flush=True)
    return records

def report(records,data,out,csv_path):
    byname={d['trace']['audio_filename']:d for d in data}
    with (out/'trace.jsonl').open('w') as tf,(out/'predictions.jsonl').open('w') as pf:
        for t in records:
            tf.write(json.dumps(t)+'\n');r=t['response'];d=byname[t['audio_filename']]
            pf.write(json.dumps(dict(audio_filename=t['audio_filename'],questions=t['questions'],question_ids=[x['question_id'] for x in d['rows']],
                predictions=[int(x) for x in r['answers']],spans=[None if s is None else [s,e] for s,e in zip(r['evidence_start'],r['evidence_end'])],latency_ms=None,error=None,timed_out=False))+'\n')
    summary=analyze(out,csv_path);changes=[];byconv={};oracles=[]
    for t in records:
        d=byname[t['audio_filename']];old=d['trace']['response'];new=t['response'];delta=[]
        for i,row in enumerate(d['rows']):
            if row['label']!='1':continue
            gold=(float(row['evidence_start']),float(row['evidence_end']))
            previous=iou(None if old['evidence_start'][i] is None else (old['evidence_start'][i],old['evidence_end'][i]),gold)
            now=iou(None if new['evidence_start'][i] is None else (new['evidence_start'][i],new['evidence_end'][i]),gold)
            cs=[c for e in t['events'] if e['stage']=='candidates' and e['index']==i for c in e['candidates']]
            oracle=max([previous]+[iou((c['start'],c['end']),gold) for c in cs])
            changes.append(dict(question_id=row['question_id'],conversation=t['audio_filename'],fold=t['exp17_fold'],baseline_tiou=previous,final_tiou=now,delta=now-previous,top5_plus_baseline_oracle=oracle))
            delta.append(now-previous);oracles.append(oracle)
        byconv[t['audio_filename']]=delta
    rng=np.random.default_rng(123);keys=list(byconv)
    bootstrap=[.6*np.mean([v for j in rng.integers(0,len(keys),len(keys)) for v in byconv[keys[j]]]) for _ in range(3000)]
    baseline=float(np.mean([x['baseline_tiou'] for x in changes]));ci=np.quantile(bootstrap,[.025,.975]).tolist()
    result=dict(**summary,baseline_tiou=baseline,delta_score=.6*(summary['mean_tiou']-baseline),paired_conversation_bootstrap_95_percent=ci,
        top5_plus_baseline_oracle_all_positives=float(np.mean(oracles)),
        recovered_zero_overlap=sum(x['baseline_tiou']==0 and x['final_tiou']>0 for x in changes),
        strong_regressed_below_half=sum(x['baseline_tiou']>=.8 and x['final_tiou']<.5 for x in changes),
        promotion_gate=summary['score']>=.8 and ci[0]>0,
        note='Grouped out-of-fold reader predictions; fixed decoder and gate. Exp15 cache/demo-bank dependence remains. Bootstrap does not correct repeated development.')
    (out/'exp17_oof.json').write_text(json.dumps(result,indent=2))
    with (out/'exp17_changes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(changes[0]));w.writeheader();w.writerows(changes)
    print(json.dumps(result,indent=2));return result

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--trace',type=Path,required=True);p.add_argument('--csv',type=Path,default=ROOT/'data/question_train.csv')
    p.add_argument('--run-name',default='exp17-oof');p.add_argument('--model',default='deepset/deberta-v3-base-squad2');p.add_argument('--revision',default='main')
    p.add_argument('--folds',type=int,default=5);p.add_argument('--epochs',type=int,default=3);p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--seed',type=int,default=1701);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--device',default='cuda')
    p.add_argument('--fit-final-if-pass',action='store_true');p.add_argument('--model-out',type=Path,default=ROOT/'models/exp17-reader')
    args=p.parse_args()
    if Path(args.run_name).name!=args.run_name:raise ValueError('Use a simple run name')
    if args.folds<2 or args.epochs<1:raise ValueError('At least two folds and one epoch required')
    out=ROOT/'reports'/args.run_name;out.mkdir(parents=True,exist_ok=False)
    import torch
    from transformers import AutoTokenizer
    from solution.reader17 import settings
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use --device cpu only if intended')
    cfg=settings();cfg['device']=args.device;cfg['batch_size']=args.batch_size
    data=load_data(args.trace,args.csv);groups=sorted({d['group'] for d in data})
    if args.folds>len(groups):raise ValueError('Too many folds')
    random.Random(args.seed).shuffle(groups);folds={g:i%args.folds for i,g in enumerate(groups)}
    if not Path(args.model).is_dir():
        from huggingface_hub import model_info
        args.revision=model_info(args.model,revision=args.revision).sha
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision,use_fast=True)
    if not tokenizer.is_fast:raise ValueError('Fast tokenizer with offset mappings required')
    items=[];projection=[]
    for d in data:
        for row in d['rows']:
            if row['label']!='1':continue
            gold=(float(row['evidence_start']),float(row['evidence_end']));oracle=0
            for feature in windows(tokenizer,row['question'],d['words'],cfg):
                target=supervision(d['words'],feature,cfg,gold);oracle=max(oracle,target['oracle'])
                items.append(dict(group=d['group'],feature=feature,target=target))
            projection.append(dict(question_id=row['question_id'],word_grid_oracle=oracle))
    if not items:raise ValueError('No training windows')
    manifest=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},config=cfg,
        input_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(),csv_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        folds={d['trace']['audio_filename']:folds[d['group']] for d in data},word_grid_oracle=float(np.mean([x['word_grid_oracle'] for x in projection])),
        versions=dict(torch=torch.__version__,transformers=__import__('transformers').__version__),tokenizer_commit=tokenizer.init_kwargs.get('_commit_hash'))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'projection.json').write_text(json.dumps(projection,indent=2))
    records=[]
    for fold in range(args.folds):
        heldout=[d for d in data if folds[d['group']]==fold];training=[x for x in items if folds[x['group']]!=fold]
        model=train_model(training,tokenizer,args,cfg,fold)
        foldrecords=evaluate_fold(model,tokenizer,heldout,cfg,args,out,fold);records.extend(foldrecords)
        (out/f'fold{fold}_predictions.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in foldrecords))
        del model;gc.collect()
        if args.device.startswith('cuda'):torch.cuda.empty_cache()
    quality=report(records,data,out,args.csv)
    if args.fit_final_if_pass and quality['promotion_gate']:
        if args.model_out.exists():raise FileExistsError('Refusing to overwrite model: '+str(args.model_out))
        model=train_model(items,tokenizer,args,cfg,100)
        args.model_out.mkdir(parents=True);model.save_pretrained(args.model_out,safe_serialization=True);tokenizer.save_pretrained(args.model_out)
        (args.model_out/'exp17_metadata.json').write_text(json.dumps(dict(kind='final',config=cfg,training_manifest=manifest,source_commit=getattr(model.config,'_commit_hash',None),oof_score=quality['score']),indent=2))
        print('Final model saved:',args.model_out)
    elif args.fit_final_if_pass:print('OOF promotion gate failed. Exp15 remains the deployed model; no final model written.')
    return 0
if __name__=='__main__':raise SystemExit(main())
