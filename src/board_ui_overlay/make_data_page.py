"""Render the quiet, fixed labels for the board's 536x960 data page."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "board_ui_overlay" / "data_page"
OUT.mkdir(parents=True, exist_ok=True)
im = Image.new("RGBA", (536, 960), (246, 246, 242, 255))
d = ImageDraw.Draw(im)
font = r"C:\Windows\Fonts\msyh.ttc"
f30 = ImageFont.truetype(font, 30)
f22 = ImageFont.truetype(font, 22)
f17 = ImageFont.truetype(font, 17)
f14 = ImageFont.truetype(font, 14)
ink = (32, 39, 37, 255)
muted = (101, 109, 104, 255)
rule = (210, 214, 207, 255)

d.text((27, 27), "环境数据", font=f30, fill=ink)
d.text((404, 38), "KEY2  返回", font=f14, fill=muted)
d.line((27, 86, 509, 86), fill=rule, width=2)
d.text((27, 105), "当前读数", font=f17, fill=muted)
d.text((27, 141), "温度  ·  SHT31", font=f14, fill=muted)
d.text((278, 141), "相对湿度  ·  SHT31", font=f14, fill=muted)
d.text((27, 230), "气压  ·  BMP280", font=f14, fill=muted)
d.text((278, 230), "主板 Wi-Fi", font=f14, fill=muted)
d.line((27, 293, 509, 293), fill=rule, width=2)

for y, name, unit, color in [
    (315, "温度趋势", "°C", (157, 105, 58, 255)),
    (548, "湿度趋势", "%RH", (66, 107, 112, 255)),
]:
    d.text((27, y), name, font=f22, fill=ink)
    d.text((469, y + 6), unit, font=f14, fill=muted)
    plot_top = y + 44
    plot_bottom = y + 188
    for j in range(3):
        yy = plot_top + (plot_bottom - plot_top) * j // 2
        d.line((45, yy, 500, yy), fill=rule, width=1)
    d.line((45, plot_top, 45, plot_bottom), fill=rule, width=1)
    d.text((45, plot_bottom + 7), "20 分钟前", font=f14, fill=muted)
    d.text((472, plot_bottom + 7), "现在", font=f14, fill=muted)
    d.line((27, y + 226, 509, y + 226), fill=rule, width=2)

d.text((27, 787), "最近报警", font=f22, fill=ink)
d.text((27, 827), "暂无记录", font=f17, fill=muted)
d.text((27, 855), "报警事件尚未接入；识别框不计作报警", font=f14, fill=muted)
d.line((27, 906, 509, 906), fill=rule, width=1)
d.text((27, 920), "数据来自 SHT31 / BMP280  ·  曲线仅使用有效采样", font=f14, fill=muted)

im.save(OUT / "base.png")
(OUT / "base.bgra").write_bytes(im.tobytes("raw", "BGRA"))

for key, label, color in [
    ("wifi_on", "已连接", (46, 110, 75, 255)),
    ("wifi_off", "未连接", (163, 111, 59, 255)),
    ("wifi_unknown", "状态未知", (101, 109, 104, 255)),
]:
    tag = Image.new("RGBA", (110, 29), (0, 0, 0, 0))
    ImageDraw.Draw(tag).text((0, 0), label, font=f17, fill=color)
    tag.save(OUT / f"{key}.png")
    (OUT / f"{key}.bgra").write_bytes(tag.tobytes("raw", "BGRA"))
