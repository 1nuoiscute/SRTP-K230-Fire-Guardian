"""Make timestamped contact sheets from the frozen one-pass boxed MP4s."""
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "model_work" / "out" / "teammate_videos_20260928"
SOURCE = BASE / "frozen_v3_5fps"
OUT = BASE / "box_review"
OUT.mkdir(parents=True, exist_ok=True)
font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 19)
for path in sorted(SOURCE.glob("*_boxed.mp4")):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened() or fps <= 0 or count <= 0:
        raise RuntimeError(path)
    duration = count / fps
    sheet = Image.new("RGB", (4 * 280, 2 * 520), (245, 245, 242))
    draw = ImageDraw.Draw(sheet)
    for j, frac in enumerate((.06, .18, .30, .42, .54, .66, .78, .90)):
        t = frac * duration
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read {path.name} at {t:.1f}s")
        pic = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pic.thumbnail((270, 480))
        col, row = j % 4, j // 4
        x = col * 280 + (280 - pic.width) // 2
        y = row * 520 + 5
        sheet.paste(pic, (x, y))
        draw.text((col * 280 + 12, row * 520 + 488), f"{t:.1f}s", font=font,
                  fill=(20, 30, 25))
    cap.release()
    sheet.save(OUT / f"{path.stem}_contact.jpg", quality=93)
    print(path.name, flush=True)
