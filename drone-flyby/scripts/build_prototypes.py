#!/usr/bin/env python3
"""K spherical-k-means prototypes per class (and background) from a DINO reference bank.

  python scripts/build_prototypes.py --bank weights/dino_bank_dinov2_vitb14.npz --k 8 \
      --output weights/dino_protos_dinov2_vitb14.npz

The bank must come from scripts/build_dino_bank.py (same encoder). Prints how tightly each class clusters and how far
apart the classes are, so weak references are visible before a run.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dtos import OBJECT_CLASSES
from drone_pipeline import prototypes as P


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bank', type=Path, default=Path('weights/dino_bank.npz')); p.add_argument('--k', type=int, default=8)
    p.add_argument('--output', type=Path); p.add_argument('--seed', type=int, default=0); p.add_argument('--quantile', type=float, default=.05)
    a = p.parse_args()
    with np.load(a.bank, allow_pickle=False) as z: feats = z['features'].astype(np.float32); labels = z['labels'].astype(int); meta = json.loads(str(z['metadata'].item()))
    if meta['classes'] != list(OBJECT_CLASSES): raise SystemExit('Bank classes differ from dtos.OBJECT_CLASSES')
    proto = P.build_prototypes(feats, labels, OBJECT_CLASSES, a.k, a.seed, a.quantile)
    out = a.output or a.bank.with_name(a.bank.stem.replace('bank', 'protos') + '.npz')
    P.save(out, proto, {'bank': a.bank.name, 'encoder_sha256': meta['encoder_sha256'], 'preprocess': meta['preprocess'], 'model': meta.get('model', 'dinov2_vits14')})
    print(f'{len(proto["centers"])} prototypes (dim {proto["centers"].shape[1]}) -> {out}')
    scorer = P.ProtoScorer(proto, OBJECT_CLASSES); names = ['background'] + list(OBJECT_CLASSES)
    own = scorer.class_max(feats)
    print(f'{"class":16s} {"refs":>5s} {"tau(p5)":>8s} {"spread":>7s}  most confusable other class (mean sim of members)')
    for i, name in enumerate(names):
        m = labels == i - 1; other = own[m].mean(axis=0).copy(); other[i] = -1
        print(f'{name:16s} {int(m.sum()):5d} {proto["tau"][i]:8.3f} {proto["spread"][i]:7.3f}  {names[int(other.argmax())]} {other.max():.3f}')


if __name__ == '__main__':
    main()
