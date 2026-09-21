#!/usr/bin/env python3
"""Tune the publisher offline: replay a saved trace (cached detections + DINO evidence, no GPU) under several
`publish:` settings and score each with the official scorer.

  python scripts/sweep_publish.py --trace results/live/<run>/trace.jsonl --config configs/fusion.yaml --scene helsinki \
      --set strict='{"min_publish":0.08,"single_sight_factor":0.4}' --set nomiss='{"miss_enabled":false}'

Needs trace.images: true (the crops are re-registered) and the ground-truth scene under src/<scene>. The camera commands in the
trace are ignored, so every variant sees exactly the same views. Prints AP50 and the number of boxes per variant.
"""
import argparse
import base64
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dtos import DroneFlybyPredictRequestDto
from drone_pipeline.config import load_config
from drone_pipeline.pipeline import Pipeline
from replay_trace import CachedDetector


def replay(rows, root, cfg):
    """Run `rows` through a fresh pipeline built from cfg; returns {frame: [prediction dicts]}."""
    cfg.trace.enabled = False; detector = CachedDetector(); pipeline = Pipeline(cfg, detector=detector); predictions = {}
    for row in rows:
        path = root / row.get('image', '__missing__')
        if not path.is_file(): raise FileNotFoundError(f"Frame {row['frame']} has no saved PNG; record with trace.images: true")
        request = json.loads(json.dumps(row['request'])); request['sequence_id'] = 'sweep'; request['request_id'] = f"sweep:{row['frame']}"
        request['view']['image'] = base64.b64encode(path.read_bytes()).decode(); detector.row = row
        response = pipeline.predict(DroneFlybyPredictRequestDto.model_validate(request))
        predictions[row['frame']] = [{'object_id': a.object_id, 'confidence': a.confidence,
                                      'bbox': (a.bbox[0] * 3840, a.bbox[1] * 2160, a.bbox[2] * 3840, a.bbox[3] * 2160)} for a in response.annotations]
    pipeline.close()
    return predictions


def sweep(rows, root, config, variants, scorer):
    out = {}
    for name, override in variants.items():
        cfg = load_config(config)
        for key, value in override.items(): setattr(cfg.publish, key, value)
        cfg.publish.mode = 'fusion'; predictions = replay(rows, root, cfg)
        out[name] = {'AP50': float(scorer(predictions)[0]), 'boxes': sum(len(v) for v in predictions.values())}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--trace', required=True); p.add_argument('--config', default='configs/fusion.yaml'); p.add_argument('--scene', default='helsinki')
    p.add_argument('--session'); p.add_argument('--set', action='append', default=[], metavar="NAME='{json overrides of publish.*}'")
    a = p.parse_args(); rows = [json.loads(x) for x in Path(a.trace).read_text().splitlines() if x]
    if a.session: rows = [r for r in rows if r['session_key'] == a.session]
    if len({r['session_key'] for r in rows}) != 1: raise SystemExit('Choose exactly one --session')
    variants = {'as_configured': {}}
    for item in a.set:
        name, _, js = item.partition('='); variants[name] = json.loads(js)
    from local_evaluator import score
    result = sweep(rows, Path(a.trace).parent, a.config, variants, lambda pred: score(a.scene, pred))
    for name, r in result.items(): print(f'{name:20s} AP50={r["AP50"]:.4f}  boxes={r["boxes"]}')


if __name__ == '__main__':
    main()
