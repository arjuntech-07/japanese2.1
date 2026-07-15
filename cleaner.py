"""
This module contains the function to process the contour in the image.
UPGRADED: Now uses local IOPaint (LaMa AI) for photorealistic text removal
and Canny Edge detection to handle both black and white text.
"""
from typing import Tuple
import cv2
import numpy as np
import requests
import base64

# This is the address of your local IOPaint server
IOPAINT_URL = "http://127.0.0.1:8080/api/v1/inpaint"


def create_smart_mask(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a mask based on edges (shapes) rather than just dark pixels.
    This fixes the bug where white text on dark backgrounds wouldn't erase.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Edge Detection: Finds the sharp outlines of the Japanese text
    edges = cv2.Canny(gray, 50, 150)

    # 2. Dilation: Expands those thin outlines into thick, solid blocks
    # to ensure the mask completely covers the text and a little bit of the background
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(edges, kernel, iterations=2)

    # 3. Find contours just so we can pass the bubble shape back to text_renderer.py
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
    else:
        # Fallback if no text is found
        h, w = image.shape[:2]
        largest_contour = np.array([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]], dtype=np.int32)

    return mask, largest_contour


def cv2_to_base64(img: np.ndarray) -> str:
    """Converts a cv2 image to a base64 string for the JSON API."""
    _, buffer = cv2.imencode('.png', img)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    # We add the data URL prefix so IOPaint knows exactly what format the text is
    return f"data:image/png;base64,{b64_str}"


def process_contour(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sends the cropped image and our smart mask to the IOPaint AI server via JSON.
    Returns the photorealistically cleaned image.
    """
    # 1. Generate the mask identifying where the text is
    mask, largest_contour = create_smart_mask(image)

    try:
        # 2. Package the data for the IOPaint API as JSON using Base64
        # This completely bypasses the FastAPI UnicodeDecodeError file bug!
        payload = {
            "image": cv2_to_base64(image),
            "mask": cv2_to_base64(mask),
        }

        # Send as JSON instead of files
        response = requests.post(IOPAINT_URL, json=payload)

        if response.status_code == 200:
            # 3. Success! Decode the AI-cleaned image back into OpenCV format
            result_bytes = np.frombuffer(response.content, np.uint8)
            cleaned_image = cv2.imdecode(result_bytes, cv2.IMREAD_COLOR)
            return cleaned_image, largest_contour
        else:
            print(f"  [!] IOPaint Error {response.status_code}. Falling back to cv2.")
            # Because we use JSON, if this fails we will FINALLY see the real error message!
            print(f"  [!] Real Error Details: {response.text}")
            return cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA), largest_contour

    except requests.exceptions.ConnectionError:
        print("  [!] IOPaint server not running. Falling back to basic cv2 smudging.")
        return cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA), largest_contour