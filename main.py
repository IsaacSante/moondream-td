from transformers import AutoModelForCausalLM
from PIL import Image
from utils.prompt import Prompt

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    device_map={"": "mps"}       # Apple-GPU
)

image = Image.open("test.jpg")

objects_of_interest = ["elephant", "corn doll", "flower doll"]  # whatever list you like
prompt   = Prompt(objects_of_interest).text  # build prompt

print(model.query(image, prompt)["answer"])
