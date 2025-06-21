# main.py  – one *separate* Moondream model per object of interest
#
# • Each object gets its own model instance (no state clashes).
# • All detections run in parallel in a ThreadPool.
# • Points are over-laid live; press **Q** to quit.

from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM
from PIL import Image
import cv2, time, itertools

# ------------------------------------------------------------------
objects_of_interest = ["dragon", "mug", "notebook"]

# auto-assign visually distinct BGR colors
COLOR_CYCLE = itertools.cycle([
    (255,   0,   0),  # blue-ish
    (  0, 255,   0),  # green-ish
    (  0,   0, 255),  # red-ish
    (255, 255,   0),  # cyan
    (255,   0, 255),  # magenta
    (  0, 255, 255),  # yellow
])
COLORS = {obj: next(COLOR_CYCLE) for obj in objects_of_interest}
# ------------------------------------------------------------------

# ---- one model per object ------------------------------------------------
MODELS = {
    obj: AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-04-14",
        trust_remote_code=True,
        device_map={"": "mps"},   # Apple-Silicon GPU
    )
    for obj in objects_of_interest
}
# -------------------------------------------------------------------------

def locate(obj_name: str, pil_img: Image.Image):
    mdl = MODELS[obj_name]
    # encode + point with this *specific* model
    encoded = mdl.encode_image(pil_img)
    pts     = mdl.point(encoded, obj_name)["points"]
    return obj_name, [(p["x"], p["y"]) for p in pts]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

MAX_WORKERS = len(objects_of_interest)

try:
    while True:
        t0 = time.perf_counter()

        ok, frame = cap.read()
        if not ok:
            continue

        # Resize so the shorter side ≤ 192 px (quick speed-up)
        scale = min(1, 192 / min(frame.shape[:2]))
        if scale < 1:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # -------- run all detections in parallel -------------------------
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            results = dict(pool.map(
                lambda o: locate(o, pil_img),
                objects_of_interest
            ))
        # ----------------------------------------------------------------

        # -------- overlay points ----------------------------------------
        for obj, pts in results.items():
            color = COLORS[obj]
            for x_norm, y_norm in pts[:10]:       # show up to 10 per object
                px = int(x_norm * frame.shape[1])
                py = int(y_norm * frame.shape[0])
                cv2.circle(frame, (px, py), 6, color, thickness=-1)
                cv2.putText(frame, obj, (px + 8, py - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        # ----------------------------------------------------------------

        print(f"{len(results)} objects processed in {time.perf_counter() - t0:.3f}s")

        cv2.imshow("Moondream points – press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
