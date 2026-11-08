"""
cleaner.py
UPGRADED AFTER VISUAL TEST:
  1. Fixed 6px border margin (instead of 8% height) so text near the top/bottom tips of tall bubbles is 100% erased!
  2. Smart Bubble Interior Fill: If text is inside a clean white/solid speech bubble, fills with pure bubble background color instead of blurry Telea smudging!
  3. Returns the exact inpaint `mask` so main.py only replaces masked text pixels (preventing overlapping bounding boxes from chopping off adjacent bubbles!).
"""
from typing import Tuple
import cv2
import numpy as np
import requests
import base64

IOPAINT_URL = "http://127.0.0.1:8085/api/v1/inpaint"


def create_smart_mask(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects text strokes inside the crop while protecting only the outer 6px padding ring.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Detect both dark text on light bubbles AND light text on dark backgrounds
    edges = cv2.Canny(gray, 45, 140)

    # Also catch solid dark ink strokes inside bright speech bubbles directly
    # (prevents hollow thick characters like on the green book cover from leaving unmasked centers)
    bg_median = np.median(gray)
    if bg_median > 180:
        _, dark_ink = cv2.threshold(gray, int(bg_median * 0.72), 255, cv2.THRESH_BINARY_INV)
        edges = cv2.bitwise_or(edges, dark_ink)

    # 2. Fixed 6px outer margin (protects the 10px YOLO padding border without missing bottom characters like ねー？)
    margin = min(6, h // 6, w // 6)
    interior_edges = np.zeros_like(edges)
    if h > 2 * margin and w > 2 * margin:
        interior_edges[margin:h - margin, margin:w - margin] = edges[margin:h - margin, margin:w - margin]
    else:
        interior_edges = edges

    # 3. Remove connected components that touch the outer margin boundary (speech bubble walls!)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(interior_edges, connectivity=8)
    filtered_edges = np.zeros_like(interior_edges)
    for lbl in range(1, num_labels):
        x, y, cw, ch, area = stats[lbl]
        touches_border = (
            x <= margin + 1 or
            y <= margin + 1 or
            (x + cw) >= (w - margin - 1) or
            (y + ch) >= (h - margin - 1)
        )
        # If a long contour touches the border, it's the bubble outline wall or panel frame -> skip it!
        if touches_border and (cw > w * 0.45 or ch > h * 0.45):
            continue
        filtered_edges[labels == lbl] = 255

    # 4. Dilate text strokes so anti-aliased fringes are completely covered
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(filtered_edges, kernel, iterations=2)

    return mask, mask


def smart_local_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Cleanly erases text without smudging dark buildings/borders into white speech bubbles.
    """
    cleaned = image.copy()

    # Find pixels just outside the text mask to sample the true local bubble background color
    dilate_ring = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    ring_mask = cv2.subtract(dilate_ring, mask)

    ring_pixels = image[ring_mask > 0]
    if len(ring_pixels) > 20:
        median_bgr = np.median(ring_pixels, axis=0).astype(np.uint8)
        std_bgr = np.std(ring_pixels, axis=0).mean()

        # If the surrounding interior is a uniform speech bubble (e.g. white/cream bubble),
        # fill directly with the clean median bubble color so zero dark smudges appear!
        if std_bgr < 28 or np.mean(median_bgr) > 220:
            cleaned[mask > 0] = median_bgr
            return cleaned

    # For textured/colored backgrounds (like book covers), use Navier-Stokes / Telea inpainting
    return cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


def cv2_to_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', img)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"


def process_contour(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (cleaned_image, text_mask).
    Note: Returning `mask` allows main.py to composite ONLY the erased text pixels
    so overlapping speech bubble boxes never chop off neighboring bubbles!
    """
    mask, _ = create_smart_mask(image)

    if cv2.countNonZero(mask) == 0:
        return image.copy(), mask

    try:
        payload = {
            "image": cv2_to_base64(image),
            "mask": cv2_to_base64(mask),
        }
        response = requests.post(IOPAINT_URL, json=payload, timeout=15)
        if response.status_code == 200:
            result_bytes = np.frombuffer(response.content, np.uint8)
            cleaned_image = cv2.imdecode(result_bytes, cv2.IMREAD_COLOR)
            if cleaned_image is not None:
                return cleaned_image, mask
    except requests.exceptions.RequestException:
        pass

    return smart_local_inpaint(image, mask), mask
  
