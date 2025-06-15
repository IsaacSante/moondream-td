from transformers import AutoModelForCausalLM
from PIL import Image

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    device_map={"": "mps"}       # Apple-GPU
)

image = Image.open("test.jpg")

result = model.caption(image, length="short")["caption"]
print(result)
