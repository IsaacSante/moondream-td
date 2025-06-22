#!/usr/bin/env python3
"""
Stream webcam frames to a remote Moondream-powered detector
and overlay the returned (x, y) points in real-time.
"""

import base64
import time
from io import BytesIO

import cv2
import numpy as np  # only used if you need to convert colour spaces
import requests

REMOTE_SERVER = "http://10.0.30.81:5001/infer"       # 👈  change if the server’s IP/port moves
JPEG_QUALITY  = 90                                   # 60-95 is a good trade-off

# ----------------------------------------------------------------------------- #
cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

try:
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            continue           # drop the frame and grab another one

        # Encode as JPEG (¼–¹⁰ the size of PNG, decodes faster too)
        success, jpeg_buf = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        if not success:
            print("[WARN] JPEG encode failed")
            continue

        image_b64 = base64.b64encode(jpeg_buf).decode()   # pure base-64, no data-URI header

        # POST to the inference server
        try:
            r = requests.post(
                REMOTE_SERVER,
                json={"image": image_b64},
                timeout=5,
            )
            r.raise_for_status()
            result = r.json()
            print("[DEBUG] from server →", result)  
        except Exception as e:
            print(f"[ERROR] inference request failed: {e}")
            result = {"objects": {}}

        # Draw the returned points
        display = frame_bgr.copy()
        h, w = display.shape[:2]

        for obj, pts in result.get("objects", {}).items():
            for p in pts:
                x = int(p["x"] * w)
                y = int(p["y"] * h)
                cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(display, obj, (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Detection", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
