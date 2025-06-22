# server.py
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM
from PIL import Image
from io import BytesIO
from utils.prompt import Prompt
from utils.percepts import Percepts

app = Flask(__name__)

# Load model and setup
objects_of_interest = ["dragon", "notebook", "mug"]
prompt = Prompt(objects_of_interest).text
percepts = Percepts(objects_of_interest)

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-06-21",
    trust_remote_code=True,
    device_map="cuda",  # uses GPU
)

print("running...")

@app.route("/infer", methods=["POST"])
def infer():
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "No image uploaded"}), 400

    image = Image.open(BytesIO(file.read()))
    raw_response = model.query(image, prompt, stream=False)["answer"]
    print(prompt)
    print(raw_response)
    res = percepts.validate_percept(raw_response)

    return jsonify(res or {"object": None, "confidence": None})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
