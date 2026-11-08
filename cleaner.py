"""
cleaner.py
UPGRADED FOR PERFECT OVAL BUBBLE PROTECTION (COLOR + B&W MANGA):
  1. Elliptical Interior Safe-Zone: Prevents rectangular YOLO boxes from spilling white paint
     over the 4 corners outside oval speech bubbles (protects hair, chins, glasses, buildings!).
  2. Direct Dark-Ink & Light-Ink Stroke Masking inside the bubble interior.
  3. Clean paper-matched background fill (matches pure white OR aged B&W manga paper tone).
"""
from typing import Tuple
import cv2
import numpy as np
import requests
import base64

IOPAINT_URL = "http://127.0.0.1:8085/api/v1/inpaint"


def create_smart_mask(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects ONLY the text characters inside the oval speech bubble interior,
    strictly ignoring the 4 outer corners of the rectangular crop and the bubble's black border.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Build an Elliptical Interior Safe Mask that matches oval manga speech bubbles!
    # This automatically excludes the 4 rectangular corners outside the oval bubble
    # (where character hair, faces, glasses, and buildings sit).
    ellipse_mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (max(4, int(w * 0.44)), max(4, int(h * 0.45)))
    cv2.ellipse(ellipse_mask, center, axes, 0, 0, 360, 255, -1)

    # 2. Sample the bubble interior brightness inside the ellipse
    interior_pixels = gray[ellipse_mask > 0]
    bg_median = float(np.median(interior_pixels)) if len(interior_pixels) > 0 else 240.0

    # 3. Detect text strokes based on whether this is a normal light bubble (B&W or Color)
    #    or an inverted dark bubble
    if bg_median >= 150:
        # Standard white / cream / aged B&W paper bubble with dark text:
        # Catch all dark ink pixels darker than 78% of the bubble's paper brightness
        thresh_val = int(bg_median * 0.78)
        _, raw_text_mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    else:
        # Dark bubble or dark background with bright/outlined text:
        edges = cv2.Canny(gray, 50, 150)
        raw_text_mask = edges

    # Restrict strictly to inside the oval ellipse safe-zone
    interior_text = cv2.bitwise_and(raw_text_mask, ellipse_mask)

    # 4. Connected-Components Filter:
    # Keep character-sized blobs; drop any long line that touches the ellipse boundary
    # (such as an overlapping neighboring bubble wall or panel line).
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(interior_text, connectivity=8)
    clean_text_mask = np.zeros_like(interior_text)

    for lbl in range(1, num_labels):
        x, y, cw, ch, area = stats[lbl]
        if area < 3:
            continue
        # If a single connected component spans more than 65% of the entire bubble width or height,
        # it is a speech bubble border line or artwork line, NOT a Japanese character!
        if cw > w * 0.65 or ch > h * 0.65:
            continue
        clean_text_mask[labels == lbl] = 255

    # 5. Moderate dilation (5x5 instead of 7x7x2) to cover anti-aliased text halos
    # without ballooning outside the bubble
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(clean_text_mask, kernel, iterations=1)

    # Final clip against the ellipse so dilation never crosses the bubble wall
    mask = cv2.bitwise_and(mask, ellipse_mask)

    return mask, mask


def smart_local_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Replaces masked text strokes with the exact local bubble paper color (for light bubbles)
    or smooth Telea inpainting (for complex colored backgrounds).
    """
    cleaned = image.copy()

    # Sample background pixels immediately surrounding the text characters
    dilate_ring = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    ring_mask = cv2.subtract(dilate_ring, mask)

    ring_pixels = image[ring_mask > 0]
    if len(ring_pixels) > 15:
        median_bgr = np.median(ring_pixels, axis=0).astype(np.uint8)
        std_bgr = np.std(ring_pixels, axis=0).mean()

        # Works for both pure white bubbles AND aged/greyish B&W manga scans!
        if std_bgr < 32 and np.mean(median_bgr) > 175:
            cleaned[mask > 0] = median_bgr
            return cleaned

    return cv2.inpaint(image, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)


def cv2_to_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', img)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"


def process_contour(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cleans text inside the bubble and returns (cleaned_image, text_mask).
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
  
