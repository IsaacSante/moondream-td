# main.py  – one model *per* object
#            • single encode reused
#            • 256-px short side
#            • models loaded in fp16  ← NEW
#
# Press **Q** in the preview window to quit.

from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM
from PIL import Image
import cv2, time, itertools
import torch                                 # ← NEW

# ------------------------------------------------------------------
objects_of_interest = ["dragon", "white book", "cup"]

COLOR_CYCLE = itertools.cycle([
    (255,   0,   0), (  0, 255,   0), (  0,   0, 255),
    (255, 255,   0), (255,   0, 255), (  0, 255, 255),
])
COLORS = {obj: next(COLOR_CYCLE) for obj in objects_of_interest}
# ------------------------------------------------------------------

# ---- one fp16 model per object -----------------------------------
MODELS = {
    obj: AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-04-14",
        trust_remote_code=True,
        device_map={"": "mps"},
        torch_dtype=torch.float16           # ← NEW (loads weights in fp16)
    )
    for obj in objects_of_interest
}
# ------------------------------------------------------------------

def locate(obj_name: str, encoded_img):
    mdl = MODELS[obj_name]
    pts = mdl.point(encoded_img, obj_name)["points"]
    return obj_name, [(p["x"], p["y"]) for p in pts]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

MAX_WORKERS     = len(objects_of_interest)
MAX_SHORT_SIDE  = 256

try:
    while True:
        t0 = time.perf_counter()

        ok, frame = cap.read()
        if not ok:
            continue

        scale = min(1, MAX_SHORT_SIDE / min(frame.shape[:2]))
        if scale < 1:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        pil_img  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # one encode, reused by all models
        encoded  = next(iter(MODELS.values())).encode_image(pil_img)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            results = dict(pool.map(
                lambda o: locate(o, encoded),
                objects_of_interest
            ))

        for obj, pts in results.items():
            color = COLORS[obj]
            for x_norm, y_norm in pts[:10]:
                px = int(x_norm * frame.shape[1])
                py = int(y_norm * frame.shape[0])
                cv2.circle(frame, (px, py), 6, color, thickness=-1)
                cv2.putText(frame, obj, (px + 8, py - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        print(f"{len(results)} objects processed in {time.perf_counter() - t0:.3f}s")
        cv2.imshow("Moondream points – press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
