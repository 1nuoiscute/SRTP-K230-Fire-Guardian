import cv2
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
PLAN = ROOT.parents[1] / 'docs' / 'FIRE_PEXELS_IGNITION_QUARANTINE_20261029.md'

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def create_new_bytes(path, data):
    with open(path, 'xb') as f:
        f.write(data)

def create_new_json(path, value):
    create_new_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))

plan_hash_before = sha256(PLAN)
if plan_hash_before.lower() != '34d4d88c748570c717d494e1d22a3feb71310f903f8b40539654e15359688608':
    raise RuntimeError('Quarantine plan hash changed; refusing pixel processing.')

sources = [
    {'id':'5657536','file':ROOT/'5657536.mp4','expected_sha256':'0A722EBD46984600D7247BD3156AF7BFFB72D0B8FF1B91B8DC471A8BD04F9B44'},
    {'id':'9685496','file':ROOT/'9685496.mp4','expected_sha256':'2FEC6C7DC9B0578E75D59D01A01B5F159F7BF4614451AD3F87CF0288D9341B96'},
]
script_hash = sha256(Path(__file__))
outputs = []
for src in sources:
    path = src['file']
    before = sha256(path)
    if before.lower() != src['expected_sha256'].lower():
        raise RuntimeError(f"Original hash mismatch before decode: {path.name}")
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV/FFmpeg could not open {path.name}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid stream metadata for {path.name}")
    indices = sorted(set(round(i * (frame_count - 1) / 11) for i in range(12)))
    frames = []
    sample_records = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            cap.release()
            raise RuntimeError(f"Could not decode frame {idx} from {path.name}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append((idx, Image.fromarray(rgb).convert('RGB')))
        sample_records.append({'frame_index_zero_based':idx,'time_seconds_estimated_from_fps':round(idx/fps,6)})
    cap.release()
    # 4 columns x 3 rows, fit each image into a 640x360 cell without cropping.
    cell_w, cell_h, label_h = 640, 360, 30
    sheet = Image.new('RGB', (cell_w*4, (cell_h+label_h)*3), 'white')
    draw = ImageDraw.Draw(sheet)
    for n, (idx, frame) in enumerate(frames):
        frame.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        x = (n % 4) * cell_w + (cell_w-frame.width)//2
        y = (n // 4) * (cell_h+label_h) + (cell_h-frame.height)//2
        sheet.paste(frame, (x,y))
        draw.text(((n%4)*cell_w+8,(n//4)*(cell_h+label_h)+cell_h+5),
                  f"frame={idx}  t~{idx/fps:.3f}s", fill='black')
    out = ROOT / f"{src['id']}_uniform_contact_sheet.jpg"
    import io
    buf = io.BytesIO()
    sheet.save(buf, format='JPEG', quality=92, optimize=False)
    create_new_bytes(out, buf.getvalue())
    after = sha256(path)
    if after.lower() != before.lower():
        raise RuntimeError(f"Original changed during decode: {path.name}")
    outputs.append({
      'video_id':src['id'],'original_file':path.name,'original_bytes':path.stat().st_size,
      'original_sha256_before_decode':before,'original_sha256_after_decode':after,
      'container_probe':'OpenCV VideoCapture with FFmpeg backend',
      'opencv_version':cv2.__version__,'reported_width':width,'reported_height':height,
      'reported_fps':fps,'reported_frame_count':frame_count,
      'duration_seconds_estimated_frame_count_div_fps':frame_count/fps,
      'sample_count':len(indices),'sampling':'12 approximately uniform frame indices including first and last reported frame',
      'samples':sample_records,'contact_sheet':out.name,'contact_sheet_sha256':sha256(out)
    })

plan_hash_after = sha256(PLAN)
if plan_hash_after.lower() != plan_hash_before.lower():
    raise RuntimeError('Quarantine plan changed during extraction.')
meta = {
 'created_utc':datetime.now(timezone.utc).isoformat(),
 'purpose':'quarantined human-only visual screening; no inference or training',
 'script_file':Path(__file__).name,'script_sha256':script_hash,
 'plan_file':str(PLAN),'plan_sha256_before_and_after':plan_hash_after,
 'software':{'python':'see local runtime','opencv':cv2.__version__,'pillow':Image.__version__},
 'notes':['Container frame count and FPS are decoder-reported metadata; duration and sample times are estimates, not verified presentation timestamps.',
          'Uniform contact sheets are sparse samples and cannot establish uninterrupted recording, physical gas-valve state, complete ignition/off event, or independent source status.',
          'No model was loaded or run.'],
 'videos':outputs
}
create_new_json(ROOT/'screening_metadata.json',meta)
print(json.dumps(meta,ensure_ascii=False,indent=2))