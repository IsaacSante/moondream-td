# main.py (Mac side)
from PIL import Image
import cv2
import time
import requests
from io import BytesIO

TD_ENDPOINT = "http://127.0.0.1:9980/percept"
REMOTE_SERVER = "http://10.0.30.81:5001/infer"
TIMEOUT = 1.0

TRIGGERS = {
    "Frog":       ["frog"],
    "Dragon":     ["green", "toy"],
    "Hourglass":  ["hour", "sand", "glass", "time"],
    "Perfume":    ["pink"],
}


def detect_object(text: str) -> str | None:
    """
    Return the object name using priority-based detection with improved specificity.
    """
    text_l = text.lower()

    # HIGHEST PRIORITY - Specific object names (these override everything)
    if "frog" in text_l:
        return "Frog"

    if "dinosaur" in text_l:
        return "Dragon"

    if any(word in text_l for word in ["hourglass", "hour", "sand", "glass", "time"]):
        return "Hourglass"

    # HIGH PRIORITY - Perfume bottles and perfume-related items
    if "perfume" in text_l or "bottle" in text_l:
        return "Perfume"

    # HIGH PRIORITY - Green figurines are Dragon (check BEFORE other detections)
    if "green" in text_l and ("figure" in text_l or "figurine" in text_l):
        return "Dragon"

    # MEDIUM PRIORITY - Pink objects are Perfume
    if "pink" in text_l:
        return "Perfume"

    # LOWER PRIORITY - More specific Dragon triggers
    if ("green" in text_l and "toy" in text_l) or "dragon" in text_l:
        return "Dragon"

    return None


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
            raw = response.json()          # adjust if your payload key is different
            resp_text = raw if isinstance(raw, str) else raw.get("object", "")
            print(resp_text)

            found = detect_object(resp_text)
            if found:
                percept = found
            else:
                percept = None
            print(f"Found {found}" if found else "None")

            payload = {"percept": percept}

            try:
                requests.post(TD_ENDPOINT, json=payload, timeout=TIMEOUT)
            except requests.RequestException as e:
                print(f"[TD POST] {e}")

        except Exception as e:
            print(f"[ERROR] Failed to get inference result: {e}")
            result = {"object": None}

        elapsed = time.perf_counter() - start

finally:
    cap.release()
