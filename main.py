# main.py (Mac side)
from PIL import Image
import cv2
import time
import requests
from io import BytesIO

TD_ENDPOINT = "http://127.0.0.1:9980/percept"
REMOTE_SERVER = "http://10.0.30.81:5001/infer"
TIMEOUT = 1.0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

try:
    while True:
        start = time.perf_counter()
        ok, frame_bgr = cap.read()
        if not ok:
            continue

        frame_bgr = cv2.resize(frame_bgr, (0, 0), fx=min(1, 192 / min(frame_bgr.shape[:2])),
                               fy=min(1, 192 / min(frame_bgr.shape[:2])))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        buf = BytesIO()
        pil_img.save(buf, format="JPEG")
        buf.seek(0)

        try:
            response = requests.post(REMOTE_SERVER, files={"image": buf}, timeout=3)
            result = response.json()
            print(result)
        except Exception as e:
            print(f"[ERROR] Failed to get inference result: {e}")
            result = {"object": None}

        elapsed = time.perf_counter() - start
        if result and result.get("object"):
            print(f"Detected: {result['object']} ({result.get('confidence')}%) | {elapsed:.3f}s")
        else:
            print(f"No object detected | {elapsed:.3f}s")

        # try:
        #     requests.post(TD_ENDPOINT, json={"percept": result["object"], "confidence": result.get("confidence")}, timeout=TIMEOUT)
        # except Exception as e:
        #     print(f"[TD POST] {e}")

finally:
    cap.release()
