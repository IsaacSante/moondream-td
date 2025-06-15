from transformers import AutoModelForCausalLM
from PIL import Image
from utils.prompt import Prompt
from utils.percepts import Percepts
import cv2
import time

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-04-14",
    trust_remote_code=True,
    device_map={"": "mps"}        # Apple-GPU
)

objects_of_interest = ["brown elephant", "yellow head doll", "pigeon"]
prompt   = Prompt(objects_of_interest).text
percepts = Percepts(objects_of_interest)

cap_num = 0 # obs camera on my system
cap = cv2.VideoCapture(cap_num )
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

try:
    while True:
        start = time.perf_counter()

        ok, frame_bgr = cap.read()
        if not ok:
            continue

        # Convert BGR → RGB → PIL
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(frame_rgb)

        res = percepts.validate_percept(
            model.query(pil_img, prompt, stream=False)["answer"]
        )

        elapsed = time.perf_counter() - start
        print(f"{res}  |  {elapsed:.3f}s")
except KeyboardInterrupt:
    pass
finally:
    cap.release()
