from transformers import AutoModelForCausalLM
from PIL import Image
from utils.prompt import Prompt
from utils.percepts import Percepts

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    device_map={"": "mps"}       # Apple-GPU
)

# setup
objects_of_interest = ["brown elephant", "yellow head doll", "pigeon"]  # whatever list you like
prompt   = Prompt(objects_of_interest).text  # build prompt
percepts = Percepts(objects_of_interest) # validation in case model returns some bs

while True:
    try: print(percepts.validate_percept(model.query(Image.open("current_frame.jpeg"), prompt)["answer"]))
    except OSError: continue
