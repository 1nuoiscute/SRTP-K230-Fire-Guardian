"""Dependency-free geometry and exposure checks for development experiments."""
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def iou(a, b):
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area_a = max(0,a[2]-a[0])*max(0,a[3]-a[1])
    area_b = max(0,b[2]-b[0])*max(0,b[3]-b[1])
    union = area_a+area_b-overlap
    return overlap/union if union else 0.0


def matches(gt, predictions, threshold=0.5):
    """One-to-one maximum-cardinality IoU matching; duplicates remain false positives."""
    edges = [[j for j,p in sorted(enumerate(predictions),key=lambda v:iou(g,v[1]),reverse=True)
              if iou(g,p) >= threshold] for g in gt]
    owners = {}
    def augment(g, seen):
        for p in edges[g]:
            if p in seen: continue
            seen.add(p)
            if p not in owners or augment(owners[p],seen):
                owners[p]=g
                return True
        return False
    for g in range(len(gt)): augment(g,set())
    return {'tp':len(owners),'fp':len(predictions)-len(owners),'fn':len(gt)-len(owners)}


def reject_exposed(path: Path, registry: Path):
    if not registry.is_file(): raise ValueError('Exposure registry missing')
    entries=json.loads(registry.read_text(encoding='utf-8'))['videos']
    digest=sha256(path)
    if any(row['sha256']==digest for row in entries):
        raise ValueError(f'Development-exposed video cannot be a fresh test: {path.name}')
    return digest


def validate_yolo(text):
    for line in text.splitlines():
        if not line.strip(): continue
        items=line.split()
        if len(items)!=5: raise ValueError('YOLO label needs five fields')
        cls,x,y,w,h=map(float,items)
        if cls!=0 or not 0<w<=1 or not 0<h<=1 or not 0<=x<=1 or not 0<=y<=1:
            raise ValueError('Invalid fire-only class or normalized box')
        if x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2>1+1e-6 or y+h/2>1+1e-6:
            raise ValueError('Box exceeds image bounds')
