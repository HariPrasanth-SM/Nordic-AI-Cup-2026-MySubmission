#!/usr/bin/env python3
"""Install on drone_flyby_v2 plus the DINO verifier overlay; refuses unknown source layouts."""
import argparse,ast,re,shutil
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
MARKER='precision-navigation-v1'

def once(text,old,new):
    if text.count(old)!=1:raise RuntimeError('Unsupported source layout; expected exactly one occurrence of '+repr(old[:100]))
    return text.replace(old,new,1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='configs/dino_verifier.yaml');p.add_argument('--output',default='configs/precision.yaml');a=p.parse_args()
    if not (ROOT/'drone_pipeline/dino_verifier.py').exists():raise SystemExit('Install the previous DINO verifier patch first.')
    changes={};path=ROOT/'drone_pipeline/pipeline.py';s=path.read_text()
    if MARKER not in s:
        s=once(s,'from .camera import CameraPolicy, legal, guard_command','from .camera import CameraPolicy, legal\nfrom .precision_camera import guard_command\nfrom .precision_gate import PrecisionGate\n# '+MARKER)
        s=once(s,'        self.camera=CameraPolicy(cfg.camera); self.cache=OrderedDict()','        self.camera=CameraPolicy(cfg.camera); self.cache=OrderedDict()\n        self.precision=PrecisionGate(cfg)')
        s=once(s,'            command,camera_info=s.camera.choose(r,s.tracker.tracks,motion,now)',
            "            exported,precision_info,focus=s.precision.process(r,detections,s.tracker.tracks,motion,now,dt,exported,getattr(self.detector,'last_info',{}))\n            command,camera_info=s.camera.choose(r,s.tracker.tracks,motion,now)\n            if focus is not None:\n                command=focus\n                camera_info['precision_revisit']=precision_info.get('revisit_track_id')")
        s=once(s,"                'motion':motion.details,'camera':camera_info,'errors':errors,","                'motion':motion.details,'camera':camera_info,'errors':errors,'precision':precision_info,")
        changes[path]=s
    path=ROOT/'drone_pipeline/camera.py';s=path.read_text()
    if '# '+MARKER not in s:
        s=once(s,'MAXIMUM_CENTER_DELTA_PIXELS[request.view.resolution_level]) - 2.)','MAXIMUM_CENTER_DELTA_PIXELS[request.view.resolution_level]) - .01)  # '+MARKER)
        changes[path]=s
    path=ROOT/'api.py';s=path.read_text()
    if '# '+MARKER not in s:
        s=once(s,'from utils import validate_response','from utils import validate_response\nfrom drone_pipeline.precision_camera import enforce_response\n# '+MARKER)
        s=once(s,'        response=predict(request)','        response=enforce_response(request,predict(request))')
        s,n=re.subn(r"^REVISION=.*$","REVISION='"+MARKER+"'",s,count=1,flags=re.M)
        if n!=1:raise RuntimeError('Missing API revision')
        changes[path]=s
    path=ROOT/'api_diagnostic.py'
    if path.exists():
        s=path.read_text()
        if '# '+MARKER not in s:
            start=s.index('def constrain(req,res):');end=s.index('class Journal:',start)
            s=s[:start]+'# '+MARKER+'\nfrom drone_pipeline.precision_camera import constrain\n\n'+s[end:]
            s,n=re.subn(r"^REVISION=.*$","REVISION='"+MARKER+"'",s,count=1,flags=re.M)
            if n!=1:raise RuntimeError('Missing diagnostic revision')
            changes[path]=s
    # Preflight syntax and configuration before touching any source.
    for path,s in changes.items():ast.parse(s,filename=str(path))
    config=yaml.safe_load((ROOT/a.base).read_text())
    config.setdefault('detector',{})['backend']='drone_pipeline.dino_verifier:create'
    config.setdefault('tracker',{})['enabled']=True
    config['export_fresh']=False
    config.setdefault('trace',{}).update(enabled=True,images=True)
    out=ROOT/a.output
    if out.resolve()==(ROOT/a.base).resolve():raise SystemExit('--output must differ from --base')
    for path,s in changes.items():
        backup=path.with_name(path.name+'.before_precision_v1')
        if not backup.exists():shutil.copy2(path,backup)
        tmp=path.with_name(path.name+'.precision_tmp');tmp.write_text(s);tmp.replace(path)
        print('Patched',path.relative_to(ROOT))
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(yaml.safe_dump(config,sort_keys=False))
    print('Config:',out.relative_to(ROOT));print('Restart the running API; check /api revision = '+MARKER)
if __name__=='__main__':main()
