"""Render the prior four training videos beside new-source representative frames."""
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "model_work" / "out" / "teammate_videos_20260928" / "source_review"
OUT.mkdir(parents=True, exist_ok=True)
paths = [ROOT / "viedos" / f"{i}.mp4" for i in range(4)]
font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 20)
sheet = Image.new("RGB", (4 * 280, 520), (245, 245, 242))
draw = ImageDraw.Draw(sheet)
for j, path in enumerate(paths):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, 12000)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(path)
    pic = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pic.thumbnail((270, 480))
    sheet.paste(pic, (j * 280 + (280 - pic.width) // 2, 5))
    draw.text((j * 280 + 12, 488), path.name, font=font, fill=(20, 30, 25))
    cap.release()
sheet.save(OUT / "old_training_videos_contact.jpg", quality=92)
