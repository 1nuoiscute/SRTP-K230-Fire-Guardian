"""Generate a clearly labelled, opaque 536x960 VO layer test asset."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "board_ui_overlay"
OUT.mkdir(parents=True, exist_ok=True)
image = Image.new("RGBA", (536, 960), (245, 245, 241, 255))
draw = ImageDraw.Draw(image)
font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 29)
small = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 20)
draw.text((28, 38), "数据页显示层测试", font=font, fill=(32, 36, 33, 255))
draw.line((28, 96, 508, 96), fill=(210, 216, 209, 255), width=2)
draw.text((28, 150), "屏幕应完全被本页盖住", font=small, fill=(55, 62, 57, 255))
draw.text((28, 192), "原摄像头和识别框不应浮在上面", font=small, fill=(55, 62, 57, 255))
draw.text((28, 840), "15 秒后自动返回原画面", font=small, fill=(95, 105, 98, 255))
image.save(OUT / "data_page_probe.png")
(OUT / "data_page_probe.bgra").write_bytes(image.tobytes("raw", "BGRA"))
