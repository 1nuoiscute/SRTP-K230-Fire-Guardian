"""Create source-only contact sheets for the five new teammate videos."""
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "viedos" / "视频" / "视频"
OUT = ROOT / "model_work" / "out" / "teammate_videos_20260928" / "source_review"
OUT.mkdir(parents=True, exist_ok=True)
paths = sorted(SOURCE.glob("*.mp4"))
font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 22)
for path in paths:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened() or fps <= 0 or total <= 0:
        raise RuntimeError(f"Cannot inspect {path}")
    duration = total / fps
    sheet = Image.new("RGB", (4 * 280, 520), (245, 245, 242))
    draw = ImageDraw.Draw(sheet)
    for j, frac in enumerate((0.1, 0.35, 0.6, 0.85)):
        t = duration * frac
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read {path.name} at {t:.1f}s")
        target = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        target.thumbnail((270, 480))
        x = j * 280 + (280 - target.width) // 2
        sheet.paste(target, (x, 5))
        draw.text((j * 280 + 10, 488), f"{t:.1f}s", fill=(20, 30, 25), font=font)
    cap.release()
    sheet.save(OUT / f"{path.stem}_contact.jpg", quality=92)
    print(path.name, round(duration, 2), flush=True)
