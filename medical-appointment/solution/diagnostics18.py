"""Gold-dependent local audit; never imported by prediction code."""
import csv,json,random
from collections import Counter,defaultdict
from pathlib import Path
from solution.local_diagnostics import iou
from solution.calibration14 import calibrate
from solution.calibration15 import extend_end
from solution.types import Span,VerifierResult

def rescue_analysis(run):
    run=Path(run);rows=list(csv.DictReader((run/'diagnosis.csv').open()))
    ts=[json.loads(x) for x in (run/'trace.jsonl').read_text().splitlines()]
    lookup={(r['conversation'],r['question']):r for r in rows}
    counts=Counter(e['stage'] for t in ts for e in t.get('events',[]))
    stats=Counter();details=[];conv=defaultdict(float);base_sum=final_sum=oracle_sum=0.
    for t in ts:
        name=t['audio_filename'].removeprefix('conversation_').removesuffix('.mp3')
        events=t['events'];b=[e for e in events if e['stage']=='exp18_baseline']
        if len(b)!=1:continue
        baseline=b[0]['results'];screen={r['index']:r for e in events if e['stage']=='exp18_screen' for r in e['cases']}
        pools={e['index']:e['candidates'] for e in events if e['stage']=='exp18_candidates'}
        cfg=next(e['calibration'] for e in events if e['stage']=='exp18_config')
        for i,q in enumerate(t['questions']):
            row=lookup[(name,q)];r=baseline[i];positive=int(row['label'])==1
            old=(r['span']['start'],r['span']['end']) if r['p_yes']>.5 and r['span'] else None
            new=(float(row['pred_start']),float(row['pred_end'])) if row['pred_start'] and int(row['prediction']) else None
            gold=(float(row['gold_start']),float(row['gold_end'])) if positive else None
            # Match public response precision for baseline comparison.
            old=tuple(round(x,2) for x in old) if old else None
            a=iou(old,gold) if positive else 0.;bscore=iou(new,gold) if positive else 0.
            eligible=i in screen;selected=eligible and screen[i]['review_selected'];changed=old!=new
            stats['classification_changes']+=int(int(row['prediction'])!=int(r['p_yes']>.5))
            if positive:
                stats['positive_observed']+=1;base_sum+=a;final_sum+=bscore;conv[name]+=.6*(bscore-a)
                stats['baseline_zero']+=int(a==0);stats['zero_eligible']+=int(a==0 and eligible)
                stats['zero_selected']+=int(a==0 and selected);stats['zero_rescued']+=int(a==0 and bscore>0)
                stats['strong_selected']+=int(a>=.5 and selected);stats['strong_damaged_below_half']+=int(a>=.5 and bscore<.5)
                stats['improved']+=int(bscore>a+1e-8);stats['regressed']+=int(bscore<a-1e-8)
                best=a
                for c in pools.get(i,[])[1:]:
                    x=VerifierResult(1,Span(c['start'],c['end']),.5,c['text'])
                    x=extend_end(calibrate([x],cfg),cfg)[0]
                    best=max(best,iou((round(x.span.start,2),round(x.span.end,2)),gold))
                oracle_sum+=best
            details.append(dict(question_id=row['question_id'],question=q,positive=positive,eligible=eligible,review_selected=selected,
                baseline_tiou=a,final_tiou=bscore,delta=bscore-a,changed=changed,baseline_quote=r.get('quote'),
                gold_text=row['gold_text'],final_text=row['pred_text'],screen=screen.get(i),candidates=pools.get(i,[]),
                review=[e for e in events if e['stage'] in ('exp18_review','exp18_confirmation','exp18_failed')]))
    denominator=sum(int(r['label']) for r in rows)
    summary=json.loads((run/'diagnosis.json').read_text())
    cov=[e for t in ts for e in t['events'] if e['stage']=='exp18_coverage']
    gate=len(ts)==summary.get('response_conversations') and summary.get('failures',1)==0 and len(cov)==len(ts) and counts['exp18_failed']==0 and stats['classification_changes']==0 and stats['positive_observed']==denominator
    # Conversation-cluster bootstrap, exploratory only; no threshold search here.
    values=list(conv.values());rng=random.Random(1801);draws=[]
    if values:
        for _ in range(4000):draws.append(sum(rng.choice(values) for _ in values)/denominator)
        draws.sort()
    report=dict(execution_gate_passed=gate,model_evaluated=counts['exp18_screen_only']==0,positive_denominator=denominator,**stats,
        baseline_tiou=base_sum/denominator,final_tiou=final_sum/denominator,
        delta_score_with_frozen_classification=.6*(final_sum-base_sum)/denominator,
        candidate_oracle_with_exp15_fallback=oracle_sum/denominator,
        eligible=sum(e['eligible'] for e in cov),selected=sum(e['selected'] for e in cov),stage_counts=dict(counts),
        paired_conversation_bootstrap_95_percent=[draws[100],draws[3899]] if draws else None,
        note='Execution is separate from quality. Exploratory bootstrap does not correct repeated tuning. Cached times measure rescue only. Candidate oracle uses gold, not achievable performance.')
    (run/'exp18_audit.json').write_text(json.dumps(report,indent=2))
    (run/'exp18_cases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in details))
    return report
