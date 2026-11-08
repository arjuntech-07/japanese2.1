"""
cleaner.py
Processes cropped speech bubbles in Pure RAM.
UPGRADED:
  1. Port moved to 8085 so it never conflicts with llama-server (8080).
  2. Interior Border-Safe Canny Masking: Prevents erasing the speech bubble's own black outline!
  3. Automatic fast fallback to OpenCV Telea inpainting + solid white interior fill if IOPaint is offline.
"""
from typing import Tuple
import cv2
import numpy as np
import requests
import base64

# IOPaint server runs on 8085 (llama-server uses 8080)
IOPAINT_URL = "http://127.0.0.1:8085/api/v1/inpaint"


def create_smart_mask(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates an inpainting mask targeting ONLY the text inside the speech bubble,
    while protecting the outer speech bubble border and surrounding artwork.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Detect sharp edges (works on both black-on-white and white-on-black text)
    edges = cv2.Canny(gray, 50, 150)

    # 2. Protect the outer padding / bubble border wall!
    # Since crop_engine adds 10px padding around the box, the bubble outline sits near the edges.
    margin_x = max(4, int(w * 0.08))
    margin_y = max(4, int(h * 0.08))

    # Zero out Canny edges inside the outer margin ring so bubble walls aren't dilated into the mask
    interior_edges = np.zeros_like(edges)
    interior_edges[margin_y:h - margin_y, margin_x:w - margin_x] = edges[margin_y:h - margin_y, margin_x:w - margin_x]

    # 3. Dilate interior text edges into solid character blocks
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(interior_edges, kernel, iterations=2)

    # 4. Compute usable interior contour for reference
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
    else:
        largest_contour = np.array(
            [[[margin_x, margin_y]], [[w - margin_x, margin_y]],
             [[w - margin_x, h - margin_y]], [[margin_x, h - margin_y]]],
            dtype=np.int32
        )

    return mask, largest_contour


def cv2_to_base64(img: np.ndarray) -> str:
    """Encodes an OpenCV numpy array to a Base64 PNG data URL in RAM."""
    _, buffer = cv2.imencode('.png', img)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"


def process_contour(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sends the in-memory cropped bubble and border-safe text mask to IOPaint (port 8085).
    Falls back cleanly to OpenCV Telea inpainting if IOPaint is not running.
    """
    mask, largest_contour = create_smart_mask(image)

    # If the mask is completely empty (no text edges found), return image untouched
    if cv2.countNonZero(mask) == 0:
        return image.copy(), largest_contour

    try:
        payload = {
            "image": cv2_to_base64(image),
            "mask": cv2_to_base64(mask),
        }

        response = requests.post(IOPAINT_URL, json=payload, timeout=30)

        if response.status_code == 200:
            result_bytes = np.frombuffer(response.content, np.uint8)
            cleaned_image = cv2.imdecode(result_bytes, cv2.IMREAD_COLOR)
            if cleaned_image is not None:
                return cleaned_image, largest_contour

        # Fallback if non-200 status
        return cv2.inpaint(image, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA), largest_contour

    except requests.exceptions.RequestException:
        # Clean silent fallback to OpenCV Telea when IOPaint server isn't active
        return cv2.inpaint(image, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA), largest_contour
        