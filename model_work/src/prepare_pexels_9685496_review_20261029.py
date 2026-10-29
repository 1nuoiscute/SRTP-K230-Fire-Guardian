import cv2
import hashlib
import io
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / 'data' / 'pexels_ignition_quarantine_20261029' / '9685496.mp4'
PLAN = ROOT / 'docs' / 'FIRE_PEXELS_9685496_SOURCE_REVIEW_20261029.md'
OLD_PLAN = ROOT / 'docs' / 'FIRE_PEXELS_IGNITION_QUARANTINE_20261029.md'
EXPECTED_PLAN = '9b9be63402e2a2881ed6ae6541c30e0d4d7a26d81b73203c94de505d35db2a04'
EXPECTED_OLD_PLAN = '34d4d88c748570c717d494e1d22a3feb71310f903f8b40539654e15359688608'
EXPECTED_SOURCE = '2fec6c7dc9b0578e75d59d01a01b5f159f7bf4614451ad3f87cf0288d9341b96'


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'xb') as f:
        f.write(data)
        f.flush()

def write_json_new(path, obj):
    write_new(path, (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))

def utcnow():
    return datetime.now(timezone.utc).isoformat()

started = utcnow()
source_hash_before = None
written = []
try:
    plan_hash = sha256_file(PLAN)
    old_plan_hash = sha256_file(OLD_PLAN)
    if plan_hash.lower() != EXPECTED_PLAN or old_plan_hash.lower() != EXPECTED_OLD_PLAN:
        raise RuntimeError('Frozen plan hash guard failed; no source decoding performed.')
    source_hash_before = sha256_file(SOURCE)
    if source_hash_before.lower() != EXPECTED_SOURCE:
        raise RuntimeError('Original source hash guard failed; no source decoding performed.')
    script_hash = sha256_file(Path(__file__))
    runtime = {
        'captured_utc': started,
        'command': 'python review_9685496.py',
        'python_version': sys.version,
        'python_executable': sys.executable,
        'platform': platform.platform(),
        'opencv_version': cv2.__version__,
        'opencv_video_capture_backend_requested': 'CAP_FFMPEG',
        'opencv_build_information_sha256': sha256_bytes(cv2.getBuildInformation().encode('utf-8')),
        'pillow_version': Image.__version__,
        'script_file': Path(__file__).name,
        'script_sha256': script_hash,
        'plan_sha256': plan_hash,
        'prior_quarantine_plan_sha256': old_plan_hash,
        'source_file': str(SOURCE),
        'source_sha256_before_decode': source_hash_before,
    }
    write_json_new(OUT / 'runtime_snapshot.json', runtime)

    cap = cv2.VideoCapture(str(SOURCE), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError('Could not open original with OpenCV FFmpeg backend.')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    reported_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    backend = cap.getBackendName()
    if abs(fps - 24.0) > 0.01 or (width, height) != (3840, 2160):
        cap.release()
        raise RuntimeError(f'Unexpected stream metadata: {width}x{height} at {fps} fps.')
    ranges = [
      {'label':'1p5_to_5s','start':36,'end':120,'step':4},
      {'label':'18_to_22s','start':432,'end':528,'step':4},
    ]
    selected = {}
    for r in ranges:
        for idx in range(r['start'], r['end'] + 1, r['step']):
            selected[idx] = r['label']
    frame_records = []
    decoded = 0
    max_index = max(selected)
    for idx in range(max_index + 1):
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise RuntimeError(f'Sequential decode ended before required frame index {idx}; decoded={decoded}.')
        decoded += 1
        if idx not in selected:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_bytes = rgb.tobytes(order='C')
        image = Image.fromarray(rgb, mode='RGB')
        rel = Path('frames') / f'frame_{idx:06d}.png'
        buf = io.BytesIO()
        image.save(buf, format='PNG', optimize=False)
        png_bytes = buf.getvalue()
        write_new(OUT / rel, png_bytes)
        written.append(str(rel))
        frame_records.append({
          'range':selected[idx], 'frame_index_zero_based':idx,
          'estimated_time_seconds_index_div_reported_fps':idx/fps,
          'width':width,'height':height,'pixel_sha256_rgb_row_major':sha256_bytes(rgb_bytes),
          'png_file':rel.as_posix(),'png_file_sha256':sha256_bytes(png_bytes),
        })
    cap.release()

    # Produce 2x2 pages; frame PNGs above remain full-resolution original decodes.
    pages = []
    cell_w, cell_h, label_h = 960, 540, 32
    for group in ranges:
        group_records = [x for x in frame_records if x['range'] == group['label']]
        for page_no, offset in enumerate(range(0, len(group_records), 4), start=1):
            batch = group_records[offset:offset+4]
            sheet = Image.new('RGB', (cell_w*2, (cell_h+label_h)*2), 'white')
            draw = ImageDraw.Draw(sheet)
            for pos, item in enumerate(batch):
                frame = Image.open(OUT / item['png_file']).convert('RGB')
                frame.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
                col, row = pos % 2, pos // 2
                x = col*cell_w + (cell_w-frame.width)//2
                y = row*(cell_h+label_h) + (cell_h-frame.height)//2
                sheet.paste(frame,(x,y))
                draw.text((col*cell_w+10,row*(cell_h+label_h)+cell_h+6),
                          f"index={item['frame_index_zero_based']}  t~{item['estimated_time_seconds_index_div_reported_fps']:.3f}s",fill='black')
            buf=io.BytesIO(); sheet.save(buf,format='JPEG',quality=95,optimize=False)
            page_rel=Path('pages')/f"{group['label']}_page_{page_no:02d}.jpg"
            page_bytes=buf.getvalue(); write_new(OUT/page_rel,page_bytes); written.append(str(page_rel))
            pages.append({'range':group['label'],'page_file':page_rel.as_posix(),
                          'frame_indices':[x['frame_index_zero_based'] for x in batch],
                          'page_sha256':sha256_bytes(page_bytes)})

    source_hash_after = sha256_file(SOURCE)
    if source_hash_after.lower() != source_hash_before.lower():
        raise RuntimeError('Original source hash changed during read-only review.')
    manifest={
      'status':'completed', 'started_utc':started,'completed_utc':utcnow(),
      'source_file':str(SOURCE),'source_bytes':SOURCE.stat().st_size,
      'source_sha256_before_decode':source_hash_before,'source_sha256_after_decode':source_hash_after,
      'plan_sha256':plan_hash,'prior_quarantine_plan_sha256':old_plan_hash,
      'reported_width':width,'reported_height':height,'reported_fps':fps,
      'reported_frame_count':reported_count,'decoder_backend':backend,
      'decode_method':'Sequential cap.read() from zero-based frame index 0 through 528; no frame seeking.',
      'decoded_frame_count':decoded,'selected_frame_count':len(frame_records),
      'ranges':ranges,'frames':frame_records,'pages':pages,
      'runtime_snapshot':'runtime_snapshot.json','runtime_snapshot_sha256':sha256_file(OUT/'runtime_snapshot.json'),
      'script_file':Path(__file__).name,'script_sha256':script_hash,
      'interpretation_limits':['Sample indices/times are based on decoder-reported constant FPS; source PTS was not verified.',
       'Sparse frames do not prove absence of edits between samples or frame-by-frame continuity.',
       'Visible flame changes do not establish physical valve state or exact event timing.',
       'No model inference or training was performed; source independence remains unreviewed.'],
    }
    write_json_new(OUT/'sampling_manifest.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
except Exception as exc:
    failure={'status':'failed','started_utc':started,'failed_utc':utcnow(),
             'error_type':type(exc).__name__,'error':str(exc),
             'source_sha256_before_decode':source_hash_before,'created_artifacts_before_failure':written,
             'script_file':Path(__file__).name,'script_sha256':sha256_file(Path(__file__))}
    try: write_json_new(OUT/'failure.json',failure)
    except FileExistsError: pass
    print(json.dumps(failure,ensure_ascii=False,indent=2),file=sys.stderr)
    raise