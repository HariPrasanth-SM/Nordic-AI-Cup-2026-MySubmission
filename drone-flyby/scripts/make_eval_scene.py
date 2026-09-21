#!/usr/bin/env python3
"""Create an official-evaluator scene containing only held-out source frames."""
import argparse,json,shutil
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path('src/helsinki')); p.add_argument('--scene',default='helsinki_holdout'); a=p.parse_args()
    if Path(a.scene).name!=a.scene: raise ValueError('Scene must be a single folder name')
    dest=Path('src')/a.scene
    if dest.exists(): raise ValueError('Use a fresh scene name')
    split=json.loads((a.dataset/'split.json').read_text())
    for sub,ext in (('images','.png'),('annotations','.json')):
        (dest/sub).mkdir(parents=True)
        for stem in split['val']: shutil.copy2(a.source/sub/(stem+ext),dest/sub/(stem+ext))
    print(f'Created {dest}: {len(split["val"])} held-out frames. This short sequence is only a local diagnostic.')
