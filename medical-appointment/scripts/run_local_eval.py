"""Local HTTP replay with durable per-conversation predictions and offline diagnosis."""
import argparse
import datetime
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import requests
import local_evaluator as official
from utils import group_questions_by_conversation,load_sample_audio,encode_audio,gold_evidence
from solution.local_diagnostics import analyze


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--profile',default='workstation_32gb')
    p.add_argument('--verifier',choices=['llm','similarity'],default='llm')
    p.add_argument('--experiment',choices=['baseline','exp10','exp11','exp12','exp13','exp14','exp15','exp16','exp17','exp18'],default='baseline')
    p.add_argument('--run-name')
    p.add_argument('--startup-timeout',type=int,default=900)
    p.add_argument('--skip-oracle',action='store_true')
    p.add_argument('--verbose',action='store_true')
    a=p.parse_args()
    os.chdir(ROOT)
    # Refuse to accidentally score an old server while the new one fails to bind.
    probe=socket.socket();occupied=probe.connect_ex(('127.0.0.1',9054))==0;probe.close()
    if occupied:raise SystemExit('Port 9054 is already occupied. Stop your existing medical server first.')
    if not a.skip_oracle and abs(official.oracle().final_score-1)>1e-9:
        raise SystemExit('Oracle score is not 1.0')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    name=a.run_name or f'{stamp}_{a.experiment}'
    if Path(name).name != name:raise SystemExit('--run-name must be a simple directory name')
    out=ROOT/'reports'/name
    out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.pop('MEDICAL_APPT_SKIP_MODEL_LOAD',None)
    env.update(MEDICAL_APPT_PROFILE=a.profile,MEDICAL_APPT_VERIFIER_KIND=a.verifier,
               MEDICAL_APPT_EXPERIMENT=a.experiment,MEDICAL_LOCAL_TRACE=str(out/'trace.jsonl'))
    try:sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:sha='nogit'
    metadata=vars(a)|{'git_commit':sha,'started_utc':stamp,
        'python':sys.executable,'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
        for folder in ['solution','configs'] for f in (ROOT/folder).rglob('*') if f.suffix in {'.py','.yaml','.json'}},
        'annotation_bank_sha256':None}
    if a.experiment == 'exp18':
        from solution.rescue18 import load_settings as rescue_settings
        metadata['exp18_rescue'] = rescue_settings()
    bank=Path(env.get('MEDICAL_ANNOTATION_BANK','models/annotation_bank.json'))
    if a.experiment in {'exp12','exp13','exp14','exp15','exp16','exp17','exp18'}:
        grounder_path=Path(env.get('MEDICAL_EXP12_CONFIG',str(ROOT/'configs/exp12_grounding.json')))
        grounder_config=json.loads(grounder_path.read_text())
        metadata['grounding_config']=grounder_config
        if a.experiment in {'exp13','exp14','exp15','exp16','exp17','exp18'}:
            ref='exp16' if a.experiment=='exp16' else 'exp13'
            cfg_path=Path(env.get('MEDICAL_'+ref.upper()+'_CONFIG',str(ROOT/('configs/'+ref+'_refinement.json'))))
            cfg13=json.loads(cfg_path.read_text())
            metadata['refinement_config']=cfg13
            if a.experiment in {'exp14','exp15','exp16','exp17','exp18'}:
                if a.experiment in {'exp15','exp16','exp17','exp18'}:
                    from solution.calibration15 import load_settings
                else:
                    from solution.calibration14 import load_settings
                metadata['calibration_config']=load_settings()
            metadata['contrast_bank_sha256']=hashlib.sha256(Path(cfg13['bank_path']).read_bytes()).hexdigest()
        bank=Path(grounder_config['bank_path'])
    if bank.exists():metadata['annotation_bank_sha256']=hashlib.sha256(bank.read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2))
    log=(out/'server.log').open('w')
    server=subprocess.Popen([sys.executable,'api.py'],cwd=ROOT,env=env,
                            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        end=time.monotonic()+a.startup_timeout
        ready=False
        while time.monotonic()<end:
            if server.poll() is not None:raise RuntimeError(f'Server exited; inspect {out}/server.log')
            try:
                ready=requests.get('http://127.0.0.1:9054/',timeout=1).status_code==200
                if ready:break
            except requests.RequestException:pass
            time.sleep(.5)
        if not ready:raise RuntimeError(f'Server startup timeout; inspect {out}/server.log')
        stats=official.Statistics();session=requests.Session()
        groups=group_questions_by_conversation();started=time.monotonic();timeouts=0;aborted=False
        with (out/'predictions.jsonl').open('w') as predictions:
            for filename,rows in groups:
                qs=[r['question'] for r in rows]
                if aborted or time.monotonic()-started>len(groups)*60:
                    aborted=True;stats.aborted=True
                    answers,spans,latency,error,timed_out=[-1]*len(rows),[None]*len(rows),None,'attempt aborted: not sent',False
                else:
                    payload=dict(audio_filename=filename,questions=qs,audio_base64=encode_audio(load_sample_audio(filename)))
                    answers,spans,latency,error,timed_out=official._ask(session,official.DEFAULT_URL,payload,len(rows))
                    stats.record_request(len(rows),latency,failed=error is not None,timed_out=timed_out)
                    timeouts=timeouts+1 if timed_out else 0
                    if timeouts>=5:aborted=True;stats.aborted=True
                record=dict(audio_filename=filename,questions=qs,question_ids=[r['question_id'] for r in rows],
                            predictions=answers,spans=spans,latency_ms=latency,error=error,timed_out=timed_out)
                predictions.write(json.dumps(record)+'\n');predictions.flush()
                for row,pred,span in zip(rows,answers,spans):
                    value=stats.record(row['question_type'],int(row['label']),pred,gold_evidence(row),span)
                    if a.verbose:print(row['question_id'],pred,value,flush=True)
                print(f'{filename}: {latency} ms; error={error}',flush=True)
        print(stats.report())
        (out/'score.txt').write_text(stats.report())
    finally:
        _terminate_process_group(server);log.close()
    result=analyze(out,ROOT/'data/question_train.csv')
    print(json.dumps(result,indent=2));print(f'Diagnostics: {out}/diagnosis.md')
    if a.experiment in {'exp12','exp13','exp14','exp15','exp16','exp17','exp18'}:
        from solution.diagnostics12 import extra_analysis
        audit=extra_analysis(out)
        if a.experiment in {'exp13','exp14','exp15','exp16','exp17','exp18'}:
            if a.experiment=='exp16':
                from solution.diagnostics16 import refinement_analysis
            else:
                from solution.diagnostics13 import refinement_analysis
            audit=refinement_analysis(out)
        if a.experiment == 'exp14':
            from solution.diagnostics14 import calibration_analysis
            audit=calibration_analysis(out)
        elif a.experiment == 'exp15':
            from solution.diagnostics15 import calibration_analysis
            audit=calibration_analysis(out)
        if a.experiment=='exp18':
            from solution.diagnostics18 import rescue_analysis
            audit=rescue_analysis(out)
        if a.experiment=='exp17':
            from solution.diagnostics17 import reader_analysis
            audit=reader_analysis(out)
        print(json.dumps(audit,indent=2))
        if not audit['execution_gate_passed']:
            print('DO NOT DEPLOY: experimental stage failed the execution/coverage gate.')
            return 2
    return 0


def _terminate_process_group(server: subprocess.Popen) -> None:
    """Kill api.py *and* whatever it spawned (the GPU worker process),
    not just the one PID -- see the start_new_session comment above.
    Falls back to plain server.terminate()/.kill() if process-group
    signaling isn't available (e.g. Windows, where start_new_session
    means something else and os.killpg doesn't exist).
    """

    try:
        pgid = os.getpgid(server.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            server.wait(timeout=10)
    except (AttributeError, ProcessLookupError, PermissionError):
        # AttributeError: no os.killpg (non-POSIX). ProcessLookupError: the
        # group is already gone -- nothing left to clean up either way.
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__=='__main__':raise SystemExit(main())
