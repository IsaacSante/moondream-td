from transformers import AutoModelForCausalLM
from PIL import Image
from utils.prompt import Prompt
from utils.percepts import Percepts
import cv2
import time
import requests

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-04-14",
    trust_remote_code=True,
    device_map={"": "mps"},
)

TD_ENDPOINT = "http://127.0.0.1:9980/percept"   # Web Server DAT URL
TIMEOUT     = 1.0                               # seconds

objects_of_interest = ["dragon", "white book", "cup"]
prompt   = Prompt(objects_of_interest).text
percepts = Percepts(objects_of_interest)

cap_num = 0
cap = cv2.VideoCapture(cap_num)
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
        pil_img  = Image.fromarray(frame_rgb)

        raw_response = model.query(pil_img, prompt, stream=False)["answer"]
        res = percepts.validate_percept(raw_response)

        elapsed = time.perf_counter() - start

        if res:
            if res['confidence'] is not None:
                print(f"Detected: {res['object']} ({res['confidence']}%)  |  {elapsed:.3f}s")
            else:
                print(f"Detected: {res['object']}  |  {elapsed:.3f}s")
            payload = {"percept": res['object'], "confidence": res.get('confidence')}
        else:
            print(f"No object detected  |  {elapsed:.3f}s")
            payload = {"percept": None}

        try:
            requests.post(TD_ENDPOINT, json=payload, timeout=TIMEOUT)
        except requests.RequestException as e:
            print(f"[TD POST] {e}")
finally:
    cap.release()
