from transformers import AutoModelForCausalLM
from PIL import Image
from utils.prompt import Prompt
from utils.percepts import Percepts

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    device_map={"": "mps"}       # Apple-GPU
)

image = Image.open("test.jpg")

# setup
objects_of_interest = ["elephant", "corn doll", "flower doll"]  # whatever list you like
prompt   = Prompt(objects_of_interest).text  # build prompt
percepts = Percepts(objects_of_interest) # validation in case model returns some bs

# model query 
model_answer = model.query(image, prompt)["answer"]
percept_of_interest = percepts.validate_percept(model_answer)
print(percept_of_interest)