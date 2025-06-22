#!/usr/bin/env python3
"""
Moondream VLM server – ONE MODEL PER OBJECT.
 • Each object gets its own AutoModelForCausalLM instance on the GPU.
 • The image is sent once, decoded once, then each model runs .point() in parallel.

NOTE: every model instance needs ~4-5 GB VRAM.  Three objects ≈ 12-15 GB.
If your GPU cannot hold that, use 4-bit weights or fewer objects.
"""

import base64
from binascii import Error as B64Error
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

from flask import Flask, jsonify, request
from PIL import Image, ImageFile
from transformers import AutoModelForCausalLM

ImageFile.LOAD_TRUNCATED_IMAGES = True
app = Flask(__name__)

# --------------------------------------------------------------------------- #
objects_of_interest = ["dragon", "notebook", "mug"]

# -- ONE MODEL PER OBJECT ---------------------------------------------------- #
MODEL_REV = "2025-06-21"          # or your preferred checkpoint
MODEL_NAME = "vikhyatk/moondream2"

model_by_object = {
    obj: AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REV,
        trust_remote_code=True,
        device_map={"": "cuda"},   # load each copy on the GPU
    )
    for obj in objects_of_interest
}

# --------------------------------------------------------------------------- #
def decode_image(b64: str) -> Image.Image:
    try:
        return Image.open(BytesIO(base64.b64decode(b64, validate=True))).convert("RGB")
    except (B64Error, Exception) as e:
        raise ValueError(f"bad image: {e}") from None


def point_with_model(obj: str, image: Image.Image):
    """
    Runs model.point() for a single object and returns (object, points[]).
    """
    model = model_by_object[obj]
    res = model.point(image, obj)          # Moondream’s built-in pointing helper
    pts = [
        p for p in res.get("points", [])
        if 0 <= p["x"] <= 1 and 0 <= p["y"] <= 1
    ]
    return obj, pts


@app.route("/infer", methods=["POST"])
def infer():
    data = request.get_json(force=True)
    if not (b64img := data.get("image")):
        return jsonify({"error": "payload must contain 'image'"}), 400

    try:
        image = decode_image(b64img)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Run every model in parallel threads
    results = {}
    with ThreadPoolExecutor(max_workers=len(objects_of_interest)) as pool:
        futures = [pool.submit(point_with_model, obj, image)
                   for obj in objects_of_interest]
        for fut in as_completed(futures):
            obj, pts = fut.result()
            if pts:
                results[obj] = pts

    return jsonify({"objects": results})


if __name__ == "__main__":
    # One HTTP request at a time keeps GPU memory deterministic.
    app.run(host="0.0.0.0", port=5001, threaded=False)
