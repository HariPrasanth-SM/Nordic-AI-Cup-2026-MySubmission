"""Gold-free candidate expansion and features for a small span-quality model."""
import math,re
import numpy as np
STOP=set('the a an is was are were did does do to of and in for it patient has have be that will should on with as at by from had been this their'.split())
def toks(s):return re.findall(r"\d+(?:\.\d+)?|[a-z]+",s.lower())
def content(s):return set(toks(s))-STOP

def build_pool(words,seeds,expand=True):
    pool={}
    def add(a,b,kind,origin):
        if not 0<=a<=b<len(words):return
        s,e=words[a].start,words[b].end
        if e<=s or e-s>25:return
        k=(round(s,5),round(e,5));text=' '.join(w.text for w in words[a:b+1])
        if k not in pool:pool[k]=dict(start=s,end=e,start_word=a,end_word=b,text=text,kinds=[],origin=origin)
        if kind not in pool[k]['kinds']:pool[k]['kinds'].append(kind)
    for seed in seeds:
        # Prefer exact endpoint proximity, not midpoint filtering, to avoid
        # accidentally including adjacent words with zero-duration timestamps.
        a=min(range(len(words)),key=lambda i:abs(words[i].start-seed['start']))
        b=min(range(a,len(words)),key=lambda i:abs(words[i].end-seed['end']))
        kind=seed.get('kind','existing');add(a,b,kind,(a,b))
        if not expand:continue
        starts={a+d for d in [-4,-2,-1,0,1,2,4]};ends={b+d for d in [-4,-2,-1,0,1,2,4]}
        for j in range(max(0,a-16),min(len(words),b+17)):
            if j==0 or re.search(r'[.!?;:]$',words[j-1].text.strip()):starts.add(j)
            if re.search(r'[.!?;:]$',words[j].text.strip()):ends.add(j)
        for lo in sorted(starts):
            for hi in sorted(ends):
                if lo<=b and hi>=a:add(lo,hi,'expanded',(a,b))
    return list(pool.values())

FEATURE_NAMES=['log_duration','log_words','duration_ratio','word_ratio','start_delta','end_delta',
    'relative_start','relative_end','question_coverage','precision','question_number_coverage',
    'has_number','has_negation','question_negation','has_daily','has_plan','has_report','has_exam',
    'start_pronoun','first_is_and','first_is_filler','ends_sentence','begins_sentence',
    'question_test','question_med','question_duration','question_plan','question_report','question_exam',
    'kind_existing','kind_core','kind_context','kind_alternative','kind_expanded','origin_start_delta','origin_end_delta',
    'contains_no_yes','question_words','commas','sentence_stops']

def features(question,words,c,baseline):
    q=content(question);ct=content(c['text']);tt=toks(c['text']);qt=toks(question)
    s,e=c['start'],c['end'];a,b=c['start_word'],c['end_word'];dur=e-s
    bs,be=(baseline['start'],baseline['end']) if baseline else (s,e)
    bd=max(.1,be-bs);bw=sum(bs<=(w.start+w.end)/2<=be for w in words)
    total=max(1,words[-1].end);nums={x for x in qt if x[0].isdigit()}
    has=lambda seq,vs:float(bool(set(seq)&set(vs.split())))
    text=c['text'].lower();prior=words[a-1].text if a else '.';oa,ob=c['origin']
    f=[math.log1p(dur),math.log1p(b-a+1),dur/bd,(b-a+1)/max(1,bw),(s-bs)/bd,(e-be)/bd,
       s/total,e/total,len(q&ct)/max(1,len(q)),len(q&ct)/max(1,len(ct)),len(nums&set(tt))/max(1,len(nums)),
       float(any(x[0].isdigit() for x in tt)),has(tt,'no not never without'),has(qt,'no not never without'),
       has(tt,'daily day days weeks week month months'),has(tt,'continue plan arrange prescribe prescription treatment'),
       has(tt,'feel felt noticed have had'),has(tt,'examination examine listening sound sounds normal'),
       has(tt[:1],'it that this those these they he she yes no'),has(tt[:1],'and but'),has(tt[:1],'well good okay now so'),
       float(bool(re.search(r'[.!?]$',text.strip()))),float(bool(re.search(r'[.!?;:]$',prior.strip()))),
       has(qt,'test tests tsh blood result results profile'),has(qt,'medicine medication drug dose treatment'),
       has(qt,'weeks week days duration long'),has(qt,'continue plan planned arrange decided'),has(qt,'report feel felt noticed'),
       has(qt,'examination examined auscultation listened'),
       *[float(k in c['kinds']) for k in ['existing','core','context','alternative','expanded']],
       a-oa,b-ob,has(tt,'yes no'),len(qt),text.count(','),len(re.findall(r'[.!?](?:\s|$)',text))]
    assert len(f)==len(FEATURE_NAMES)
    return f

def matrix(question,words,pool,baseline):return np.asarray([features(question,words,c,baseline) for c in pool],dtype=np.float32)
