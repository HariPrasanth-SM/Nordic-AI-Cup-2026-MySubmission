#!/usr/bin/env python3
"""Per-class object size statistics in SOURCE pixels, from the training labels (no images needed).

  python scripts/build_size_prior.py --dataset datasets/helsinki_balanced_v2 --output weights/size_prior.json \
      [--annotations src/helsinki/annotations]

YOLO labels are normalized to the 480x270 detector tile; a tile pixel is a view pixel, and one L1 view pixel is 2
source pixels (L2 is native). Boxes touching the tile border are truncated objects and are skipped, so are
boxes touching the frame border in the optional real annotations (source-pixel XYXY JSON files).
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dtos import OBJECT_CLASSES, IMAGE_WIDTH, IMAGE_HEIGHT
from drone_pipeline.priors import class_stats

TILE = (480, 270); SCALE = {'L1': 2., 'L2': 1.}


def from_labels(dataset, levels, splits, edge=.5):
    sizes = {c: [] for c in OBJECT_CLASSES}; used = 0
    for level in levels:
        for split in splits:
            for path in sorted((dataset / level / 'labels' / split).glob('*.txt')):
                for line in path.read_text().splitlines():
                    parts = line.split()
                    if len(parts) != 5: continue
                    c = int(parts[0]); cx, cy, w, h = (float(v) for v in parts[1:])
                    x1, x2 = (cx - w / 2) * TILE[0], (cx + w / 2) * TILE[0]; y1, y2 = (cy - h / 2) * TILE[1], (cy + h / 2) * TILE[1]
                    if x1 <= edge or y1 <= edge or x2 >= TILE[0] - edge or y2 >= TILE[1] - edge: continue
                    sizes[OBJECT_CLASSES[c]].append((w * TILE[0] * SCALE[level], h * TILE[1] * SCALE[level])); used += 1
    return sizes, used


def from_annotations(folder, sizes, edge=2.):
    used = 0
    for path in sorted(Path(folder).glob('*.json')):
        for a in json.loads(path.read_text())['annotations']:
            x1, y1, x2, y2 = a['bbox']
            if x1 <= edge or y1 <= edge or x2 >= IMAGE_WIDTH - edge or y2 >= IMAGE_HEIGHT - edge: continue
            sizes[a['object_id']].append((x2 - x1, y2 - y1)); used += 1
    return used


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', type=Path, default=Path('datasets/helsinki_balanced_v2')); p.add_argument('--levels', nargs='+', default=['L1', 'L2'])
    p.add_argument('--splits', nargs='+', default=['train']); p.add_argument('--annotations', type=Path)
    p.add_argument('--output', type=Path, default=Path('weights/size_prior.json'))
    a = p.parse_args()
    sizes, used = from_labels(a.dataset, a.levels, a.splits); real = from_annotations(a.annotations, sizes) if a.annotations else 0
    stats = {c: class_stats(np.array(v) if v else np.empty((0, 2))) for c, v in sizes.items()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({'version': 1, 'classes': list(OBJECT_CLASSES), 'source': {'label_boxes': used, 'annotation_boxes': real}, 'stats': stats}, indent=1))
    print(f'{used} label boxes + {real} annotation boxes -> {a.output}')
    for c, s in stats.items():
        print(f"  {c:16s} n={s['n']:5d}" + ('' if s['n'] == 0 else f"  median {s['size_px_median']:6.1f}px  p05-p95 {s['size_px_p05']:6.1f}-{s['size_px_p95']:6.1f}"))
    missing = [c for c, s in stats.items() if s['n'] < 5]
    if missing: print('WARNING: too few boxes, size prior disabled for:', ', '.join(missing))


if __name__ == '__main__':
    main()
