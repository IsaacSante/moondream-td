#!/usr/bin/env python3
"""
Client – capture webcam, send frame, draw returned points.
Press Q to quit.
"""

import base64
import cv2
import requests

REMOTE_SERVER = "http://10.0.30.81:5001/infer"
JPEG_QUALITY  = 90
TIMEOUT_S     = 60            # allow all three models time to finish
WEBCAM_IDX    = 0

# --------------------------------------------------------------------------- #
cap = cv2.VideoCapture(WEBCAM_IDX, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            continue

        payload = {"image": base64.b64encode(buf).decode()}
        try:
            r = requests.post(REMOTE_SERVER, json=payload, timeout=TIMEOUT_S)
            r.raise_for_status()
            result = r.json()
            print(result)
        except Exception as e:
            print("[ERROR] inference request failed:", e)
            result = {"objects": {}}

        # draw
        disp = frame.copy()
        h, w = disp.shape[:2]
        for obj, pts in result.get("objects", {}).items():
            for p in pts:
                x, y = int(p["x"] * w), int(p["y"] * h)
                cv2.circle(disp, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(disp, obj, (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("Moondream multi-model detection", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
