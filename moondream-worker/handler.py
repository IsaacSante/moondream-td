import io, base64
from PIL import Image
from transformers import AutoModelForCausalLM
import runpod

# load model
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-06-21",
    trust_remote_code=True,
    device_map="cuda"          # Runpod attaches a GPU automatically
)

def handler(job):
    """
    job["input"] must be:
      {
        "image_b64": "<JPEG-bytes base64>",
        "prompt":    "<question for the image>"
      }
    """
    b64_img   = job["input"]["image_b64"]
    question  = job["input"]["prompt"]

    img = Image.open(io.BytesIO(base64.b64decode(b64_img)))
    answer = model.query(img, question, stream=False)["answer"]
    return {"answer": answer}

runpod.serverless.start({"handler": handler})  
