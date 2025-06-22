import base64, re
from io import BytesIO
from binascii import Error as B64Error
from typing import Dict, List

from flask import Flask, jsonify, request
from PIL import Image, ImageFile
from transformers import AutoModelForCausalLM

ImageFile.LOAD_TRUNCATED_IMAGES = True
app = Flask(__name__)

objects_of_interest = ["dragon", "notebook", "mug"]

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-06-21",
    trust_remote_code=True,
    device_map="cuda",
)

# --- regex that finds pairs like [0.23,0.51] or 0.23, 0.51 ---
COORD_RE = re.compile(r"(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)")

def query_points(image: Image.Image, obj: str) -> List[Dict[str, float]]:
    prompt = (
        f"Return the x,y coordinates (0-1 normalised) of every {obj} in the image. "
        "If none, reply 'none'."
    )
    answer = model.query(image, prompt, stream=False)["answer"]
    print(f"[DEBUG] {obj} → {answer!r}")          # 👈  see exactly what Moondream says

    answer_low = answer.lower()
    if "none" in answer_low:
        return []

    pts = []
    for x_str, y_str in COORD_RE.findall(answer):
        try:
            x, y = float(x_str), float(y_str)
            if 0 <= x <= 1 and 0 <= y <= 1:
                pts.append({"x": x, "y": y})
        except ValueError:
            pass
    return pts

@app.route("/infer", methods=["POST"])
def infer():
    data = request.get_json(force=True)
    b64img = data.get("image")
    if not b64img:
        return jsonify({"error": "No image field"}), 400

    try:
        img_bytes = base64.b64decode(b64img, validate=True)
        image     = Image.open(BytesIO(img_bytes)).convert("RGB")
    except (B64Error, Exception) as e:
        return jsonify({"error": f"bad image: {e}"}), 400

    results = {}
    for obj in objects_of_interest:
        pts = query_points(image, obj)
        if pts:
            results[obj] = pts

    print("[DEBUG] results →", results)           # 👈  final JSON going back
    return jsonify({"objects": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=False)
