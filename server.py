from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM
from PIL import Image
from io import BytesIO
import base64
import concurrent.futures

app = Flask(__name__)

objects_of_interest = ["dragon", "notebook", "mug"]

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-06-21",
    trust_remote_code=True,
    device_map="cuda",
)

def get_points_for_object(image, obj):
    prompt = f"Return the x,y coordinates (0-1 normalized) of all {obj} in the image. If none found, say 'none'."
    response = model.query(image, prompt, stream=False)["answer"]
    
    # Basic parsing - adjust based on actual model response format
    points = []
    if "none" not in response.lower():
        # Parse coordinates from response (this needs adjustment based on actual format)
        # For now, returning mock point
        points.append({"x": 0.5, "y": 0.5})
    
    return obj, points

@app.route("/infer", methods=["POST"])
def infer():
    data = request.get_json()
    image_data_uri = data["image"]
    image_base64 = image_data_uri.split(",")[1]
    image = Image.open(BytesIO(base64.b64decode(image_base64)))
    
    # Parallel calls for each object
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(objects_of_interest)) as executor:
        futures = [executor.submit(get_points_for_object, image, obj) for obj in objects_of_interest]
        
        for future in concurrent.futures.as_completed(futures):
            obj_name, points = future.result()
            if points:
                results[obj_name] = points
    
    return jsonify({"objects": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
