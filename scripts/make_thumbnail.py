"""Tạo docs/thumbnail.png (1280×720) cho dự án M31. Không dùng ảnh BDD100K (không được phân phối lại).

Chạy:  .venv\\Scripts\\python.exe scripts\\make_thumbnail.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "thumbnail.png"
FONTS = Path("C:/Windows/Fonts")

ASPHALT, LANE, WHITE = (34, 39, 46), (242, 183, 5), (255, 255, 255)
SOFT, STEEL = (213, 218, 224), (108, 142, 164)


def font(name: str, size: int, fallback: str = "arial.ttf") -> ImageFont.FreeTypeFont:
    for n in (name, fallback):
        try:
            return ImageFont.truetype(str(FONTS / n), size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    W, H = 1280, 720
    im = Image.new("RGB", (W, H), ASPHALT)
    d = ImageDraw.Draw(im)

    # Hoạ tiết: vạch kẻ đường + một box và một mask cách điệu bên phải.
    for i in range(6):
        d.rectangle([60 + i * 205, 640, 60 + i * 205 + 115, 652], fill=LANE)
    d.rounded_rectangle([880, 95, 1210, 325], radius=8, outline=LANE, width=6)
    d.rounded_rectangle([912, 135, 1178, 295], radius=40, fill=STEEL)
    d.rounded_rectangle([960, 410, 1150, 540], radius=6, outline=SOFT, width=4)
    d.text((888, 55), "car 0.84", font=font("calibrib.ttf", 26), fill=LANE)
    d.text((965, 548), "car 0.31 · review?", font=font("calibri.ttf", 24), fill=SOFT)

    d.text((60, 150), "M31", font=font("cambriab.ttf", 120), fill=LANE)
    d.text((60, 290), "Auto-Label Correction Estimator", font=font("cambriab.ttf", 50), fill=WHITE)
    d.text((60, 370), "Ước lượng ảnh nào cần review và tốn bao lâu để sửa nhãn YOLO / SAM",
           font=font("calibri.ttf", 28), fill=SOFT)
    d.text((60, 470), "Pilot 100 ảnh BDD100K · lớp car", font=font("calibrib.ttf", 30), fill=WHITE)
    d.text((60, 515), "Kết quả trung thực: chưa tốt hơn baseline “luôn đoán 0 giây”",
           font=font("calibri.ttf", 28), fill=LANE)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    im.save(OUT, optimize=True)
    print(f"Đã ghi {OUT}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
