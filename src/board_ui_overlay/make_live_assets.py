"""Generate the board-only UI backgrounds and bitmap digits (no browser on board)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parents[1] / "artifacts" / "board_ui_overlay" / "live"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
FONT = Path(r"C:\Windows\Fonts\segoeuib.ttf")
WIDTH, HEIGHT = 536, 348
CHARS = "0123456789.-"
BG = (245, 245, 241, 255)
FG = (32, 36, 33, 255)


def save_bgra(image: Image.Image, path: Path) -> None:
    path.write_bytes(image.convert("RGBA").tobytes("raw", "BGRA"))


def render_base(name: str, wifi: str, color: str) -> dict:
    html = (ROOT / "panel_template.html").read_text(encoding="utf-8")
    html = html.replace('<span class="unit">°C</span>', '')
    html = html.replace('<span class="unit">%</span>', '')
    for key, value in {
        "TEMP": "",
        "HUMIDITY": "",
        "PRESSURE": '<span style="display:inline-block;width:90px"></span>',
        "WIFI": wifi,
        "WIFI_COLOR": color,
    }.items():
        html = html.replace("{{" + key + "}}", value)
    html = html.replace("</span> hPa</strong>", "</span></strong>")
    html_path = OUT / f"base_{name}.html"
    png_path = OUT / f"base_{name}.png"
    raw_path = OUT / f"base_{name}.bgra"
    html_path.write_text(html, encoding="utf-8")
    profile = Path(tempfile.gettempdir()) / "srtp-live-ui-edge"
    subprocess.run([str(EDGE), "--headless", "--disable-gpu", "--no-first-run",
                    f"--user-data-dir={profile}", f"--window-size={WIDTH},{HEIGHT}",
                    f"--screenshot={png_path}", html_path.as_uri()],
                   check=True, capture_output=True, timeout=20)
    with Image.open(png_path) as source:
        image = source.convert("RGBA").crop((0, 0, WIDTH, HEIGHT))
        save_bgra(image, raw_path)
    return {"file": raw_path.name, "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()}


def glyph_atlas(name: str, font_size: int, cell_w: int, cell_h: int, top: int) -> dict:
    font = ImageFont.truetype(str(FONT), font_size)
    image = Image.new("RGBA", (cell_w * len(CHARS), cell_h), BG)
    draw = ImageDraw.Draw(image)
    widths = []
    for index, char in enumerate(CHARS):
        draw.text((index * cell_w, top), char, font=font, fill=FG)
        widths.append(round(draw.textlength(char, font=font)))
    png_path = OUT / f"glyphs_{name}.png"
    raw_path = OUT / f"glyphs_{name}.bgra"
    image.save(png_path)
    save_bgra(image, raw_path)
    return {"file": raw_path.name, "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "cell_w": cell_w, "cell_h": cell_h, "widths": widths}


def unit_sprite(name: str, label: str, font_size: int, width: int, height: int, top: int) -> dict:
    font = ImageFont.truetype(str(Path(r"C:\Windows\Fonts\segoeui.ttf")), font_size)
    image = Image.new("RGBA", (width, height), BG)
    ImageDraw.Draw(image).text((0, top), label, font=font, fill=FG)
    raw_path = OUT / f"unit_{name}.bgra"
    save_bgra(image, raw_path)
    image.save(OUT / f"unit_{name}.png")
    return {"file": raw_path.name, "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "width": width, "height": height}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "BGRA8888", "panel": [WIDTH, HEIGHT], "characters": CHARS,
        "bases": [
            render_base("off", "未连接", "#a87946"),
            render_base("on", "已连接", "#587d61"),
            render_base("unknown", "状态未知", "#8b928b"),
        ],
        "large": glyph_atlas("large", 28, 24, 40, -3),
        "small": glyph_atlas("small", 13, 14, 20, -1),
        "units": {
            "temp": unit_sprite("temp", "°C", 16, 34, 25, -2),
            "humidity": unit_sprite("humidity", "%", 16, 18, 25, -2),
            "pressure": unit_sprite("pressure", "hPa", 13, 28, 20, -1),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
