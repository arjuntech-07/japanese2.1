"""
text_renderer.py
Renders translated English text into a cleaned manga/manhwa speech bubble in Pure RAM.
UPGRADED:
  1. True pixel-width word wrapping (never clips off left/right bubble edges).
  2. Checks both horizontal width and vertical height when auto-sizing font.
  3. Auto-detects Dark/Black speech bubbles and flips to crisp White text with a Black halo!
  4. Auto-fetches a clean comic font into /tmp/fonts if no local manga font is mounted.
"""
import os
import urllib.request
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

FONT_CACHE_DIR = "/tmp/fonts"
os.makedirs(FONT_CACHE_DIR, exist_ok=True)


def ensure_fallback_comic_font() -> str:
    """Downloads a clean open-source comic/manga font to /tmp/fonts once if no local font exists."""
    cached_font = os.path.join(FONT_CACHE_DIR, "ComicNeue-Bold.ttf")
    if os.path.exists(cached_font) and os.path.getsize(cached_font) > 10000:
        return cached_font

    url = "https://github.com/google/fonts/raw/main/ofl/comicneue/ComicNeue-Bold.ttf"
    try:
        urllib.request.urlretrieve(url, cached_font)
        if os.path.exists(cached_font):
            return cached_font
    except Exception:
        pass
    return ""


def get_font_path(style: str = "regular") -> str:
    """Finds the best available manga/comic TTF font."""
    candidates = [
        "./fonts/fonts_animeace_i.ttf",
        "./fonts/mangat.ttf",
        "./fonts/NotoSans-Bold.ttf",
        "/kaggle/input/datasets/rohith7boi/manga-fonts/fonts/fonts_animeace_i.ttf",
        "/kaggle/input/datasets/rohith7boi/manga-fonts/fonts/mangat.ttf",
        "/kaggle/input/datasets/rohith7boi/manga-fonts/fonts/NotoSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    dl_font = ensure_fallback_comic_font()
    if dl_font:
        return dl_font

    sys_fallbacks = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    for path in sys_fallbacks:
        if os.path.exists(path):
            return path

    return ""


def wrap_text_pixel_width(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, max_width: int) -> list:
    """
    Wraps text into lines by measuring exact pixel width with `draw.textlength`.
    Guarantees lines wrap before exceeding `max_width`.
    """
    words = text.split()
    if not words:
        return []

    lines = []
    current_line = words[0]

    for word in words[1:]:
        test_line = f"{current_line} {word}"
        if draw.textlength(test_line, font=font) <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)
    return lines


def add_text(image: np.ndarray, text: str):
    """
    Draws centered, outlined English comic text inside the in-memory `image` (numpy BGR array).
    Automatically uses black text on light bubbles and white text on dark/black bubbles.
    Modifies `image` in-place.
    """
    if not text or not text.strip():
        return

    h, w = image.shape[:2]
    if h < 16 or w < 16:
        return

    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    # Safe interior padding (12% horizontal/vertical so text stays comfortably inside bubbles)
    pad_x = max(6, int(w * 0.12))
    pad_y = max(6, int(h * 0.12))
    avail_w = max(12, w - 2 * pad_x)
    avail_h = max(12, h - 2 * pad_y)

    clean_text = text.strip()
    if clean_text.startswith("*") and clean_text.endswith("*") and len(clean_text) > 2:
        clean_text = clean_text.strip("*").strip()

    font_path = get_font_path()

    def load_font(sz: int):
        if font_path:
            try:
                return ImageFont.truetype(font_path, sz)
            except Exception:
                pass
        return ImageFont.load_default()

    # Starting and minimum font sizes proportional to bubble dimensions
    max_start_size = max(14, min(36, int(min(w, h) * 0.20)))
    min_size = 9
    font_size = max_start_size

    font = load_font(font_size)
    lines = [clean_text]
    line_height = int(font_size * 1.22)

    # Iteratively shrink font_size until BOTH width and height fit inside (avail_w, avail_h)
    while font_size >= min_size:
        font = load_font(font_size)
        line_height = max(10, int(font_size * 1.22))
        lines = wrap_text_pixel_width(clean_text, draw, font, avail_w)

        total_h = len(lines) * line_height
        max_line_w = max((draw.textlength(ln, font=font) for ln in lines), default=0)

        if total_h <= avail_h and max_line_w <= avail_w:
            break

        font_size -= 1

    # Auto-detect Dark/Black speech bubbles and invert text/halo colors
    center_crop = cv2.cvtColor(image[h // 4: 3 * h // 4, w // 4: 3 * w // 4], cv2.COLOR_BGR2GRAY)
    is_dark_bubble = (float(np.median(center_crop)) < 115.0) if center_crop.size > 0 else False

    if is_dark_bubble:
        text_col = (248, 248, 248)     # Crisp white text for dark/black bubbles
        outline_col = (15, 15, 15)     # Dark black halo outline
    else:
        text_col = (15, 15, 15)        # Crisp black text for light bubbles
        outline_col = (255, 255, 255)  # Crisp white halo outline

    stroke_width = max(2, min(4, font_size // 10))

    total_block_h = len(lines) * line_height
    start_y = pad_y + max(0, (avail_h - total_block_h) // 2)

    y = start_y
    for ln in lines:
        lw = draw.textlength(ln, font=font)
        x = pad_x + max(0, (avail_w - lw) / 2.0)

        try:
            draw.text(
                (x, y),
                ln,
                fill=text_col,
                font=font,
                stroke_width=stroke_width,
                stroke_fill=outline_col
            )
        except TypeError:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                draw.text((x + dx, y + dy), ln, fill=outline_col, font=font)
            draw.text((x, y), ln, fill=text_col, font=font)

        y += line_height

    # Copy rendered result back into the OpenCV BGR numpy array in-place
    image[:, :, :] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
  
