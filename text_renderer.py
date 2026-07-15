"""
Add translated text to a cropped manga region.
OPTIMIZED: Larger text, better fitting, proper font fallbacks
"""
import unicodedata
import textwrap
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2


def detect_script(text: str) -> str:
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            if "LATIN" in name:
                return "Latin"
    return "Latin"


def get_font_path(script: str, style="regular"):
    # Prioritize manga-style fonts that actually exist in your directory
    manga_fonts = {
        "regular": [
            "/kaggle/input/manga-fonts/fonts/fonts_animeace_i.ttf",
            "/kaggle/input/manga-fonts/fonts/mangat.ttf",
            "/kaggle/input/manga-fonts/fonts/NotoSans-Regular.ttf",
        ],
        "bold": [
            "/kaggle/input/manga-fonts/fonts/fonts_animeace_i.ttf",
        ],
        "italic": [
            "/kaggle/input/manga-fonts/fonts/fonts_animeace_i.ttf",
        ]
    }

    # System font fallbacks
    fallbacks = [
        "/System/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",

        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "./fonts/NotoSans-Regular.ttf",

        "/System/Library/Fonts/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",

        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "./fonts/NotoSans-Bold.ttf",
    ]

    for f in manga_fonts.get(style, []) + fallbacks:
        if os.path.exists(f):
            return f

    try:
        return ImageFont.load_default()
    except:
        return None


def truncate_words(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    words = text.split()
    out = ""
    for w in words:
        if len(out) + len(w) + 1 > max_chars:
            break
        out += (" " if out else "") + w
    return out.rstrip() + "…"


def add_text(image: np.ndarray, text: str):
    if not text or not text.strip():
        return

    h, w = image.shape[:2]
    script = detect_script(text)

    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    pad_x = int(w * 0.05)
    pad_y = int(h * 0.05)
    avail_w = max(10, w - 2 * pad_x)
    avail_h = max(10, h - 2 * pad_y)

    is_vertical = h > w * 1.2

    style = "regular"
    t = text.strip()
    if t.endswith("!") or "!!" in t:
        style = "bold"
    if t.startswith("*") and t.endswith("*"):
        style = "italic"
        t = t.strip("*")
    text = t

    base_size = max(22, int(min(w, h) * 0.16))
    min_size = max(8, int(min(w, h) * 0.04))
    max_size = int(min(w, h) * 0.12)

    font_path = get_font_path(script, style)

    # ✅ DEBUG LINE ADDED (as requested)
    print(f"   📝 Font: {font_path}, Size: {base_size}, Bubble: {w}x{h}")

    font_size = base_size

    def load_font(sz):
        if font_path:
            try:
                return ImageFont.truetype(font_path, sz)
            except:
                pass
        try:
            return ImageFont.load_default()
        except:
            return None

    font = load_font(font_size)
    if font is None:
        return

    line_height = int(font_size * 1.3)

    if is_vertical:
        chars = max(6, int(avail_w / (font_size * 0.6)))
        wrapped = textwrap.fill(text, width=chars, break_long_words=False)
    else:
        wrap = max(8, int(avail_w / (font_size * 0.5)))
        wrapped = textwrap.fill(text, width=wrap, break_long_words=False)

    lines = wrapped.split("\n")

    for iteration in range(30):
        total_h = len(lines) * line_height

        if total_h <= avail_h:
            break

        if font_size <= min_size:
            if len(lines) > 1:
                lines = lines[:-1]
                lines[-1] = lines[-1] + "…"
            break

        font_size -= 1
        if font_size < min_size:
            font_size = min_size

        font = load_font(font_size)
        line_height = int(font_size * 1.3)

        if is_vertical:
            chars = max(6, int(avail_w / (font_size * 0.6)))
            wrapped = textwrap.fill(text, width=chars, break_long_words=False)
        else:
            wrap = max(8, int(avail_w / (font_size * 0.5)))
            wrapped = textwrap.fill(text, width=wrap, break_long_words=False)

        lines = wrapped.split("\n")

    text_col = (0, 0, 0)
    outline = (255, 255, 255)
    # Thinner outline: divided by 25 instead of 14, capped at max 3 pixels
    stroke = max(1, min(3, font_size // 25))

    def draw_line(x, y, s):
        for adj in range(1, stroke + 1):
            for dx, dy in [(-adj, 0), (adj, 0), (0, -adj), (0, adj),
                           (-adj, -adj), (adj, -adj), (-adj, adj), (adj, adj)]:
                draw.text((x + dx, y + dy), s, outline, font)
        draw.text((x, y), s, text_col, font)

    if is_vertical:
        x = w // 2
        y = pad_y + (avail_h - len(lines) * line_height) // 2
        for ln in lines:
            lw = draw.textlength(ln, font)
            draw_line(x - lw // 2, y, ln)
            y += line_height
    else:
        y = pad_y + (avail_h - len(lines) * line_height) // 2
        for ln in lines:
            lw = draw.textlength(ln, font)
            draw_line(pad_x + (avail_w - lw) // 2, y, ln)
            y += line_height

    image[:, :, :] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)