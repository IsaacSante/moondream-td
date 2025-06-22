#!/usr/bin/env python3
"""
Flask micro-service that accepts a base-64 JPEG, runs Moondream2 on it,
and returns normalised (x, y) points for each object of interest.
"""

import base64
from binascii import Error as B64Error
from io import BytesIO
from typing import Dict, List

from flask import Flask, jsonify, request
from PIL import Image, ImageFile
from transformers import AutoModelForCausalLM  # or your specialised wrapper

# Let Pillow load slightly truncated JPEGs instead of aborting
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)

# ---------------------------------------  ML SET-UP  ------------------------- #
objects_of_interest = ["dragon", "notebook", "mug"]

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-06-21",
    trust_remote_code=True,
    device_map="cuda",          # "mps" for Apple, "" for CPU
)

# Depending on the fork you use there may be a .generate, .predict, or .query;
# adapt this helper accordingly.
def query_points(image: Image.Image, obj: str) -> List[Dict[str, float]]:
    """
    Ask the model for (x,y) coords of every *obj* in *image*.
    Returns an empty list if none found.
    """
    prompt = (
        f"Return the x,y coordinates (0-1 normalised) of all {obj} in the image. "
        "If none found, say 'none'."
    )
    answer = model.query(image, prompt, stream=False)["answer"].lower()

    if "none" in answer:
        return []

    # 👉  **TODO:** replace this with real parsing once you control the model output
    # Expect something like  "[0.23,0.51];[0.77,0.12]"
    pts = []
    for token in answer.replace("[", "").split("]"):
        if "," in token:
            try:
                x_str, y_str = token.split(",")[:2]
                pts.append({"x": float(x_str), "y": float(y_str)})
            except ValueError:
                pass
    return pts


# ---------------------------------------  API  ------------------------------ #
@app.route("/infer", methods=["POST"])
def infer():
    data = request.get_json(force=True)

    b64_img = data.get("image")
    if not b64_img:
        return jsonify({"error": "Payload must include an 'image' field"}), 400

    # Robust base-64 decode
    try:
        img_bytes = base64.b64decode(b64_img, validate=True)
    except B64Error as e:
        return jsonify({"error": f"Invalid base64: {e}"}), 400

    # Decode into a Pillow RGB image
    try:
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"PIL could not open image: {e}"}), 400

    # Call the model once per object (serial is safer than multi-thread with CUDA)
    results = {}
    for obj in objects_of_interest:
        pts = query_points(image, obj)
        if pts:
            results[obj] = pts

    return jsonify({"objects": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=False)   # threaded=False –> 1 request at a time
