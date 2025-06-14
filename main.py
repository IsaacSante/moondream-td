import moondream as md
from PIL import Image
# Initialize for Moondream Local Server
model = md.vl(endpoint="http://localhost:2020/v1")
# Load an image
image = Image.open("test.jpg")
# Ask a question
answer = model.query(image, "What's in this image?")["answer"]
print("Answer:", answer)

# Stream the response
# for chunk in model.caption(image, stream=True)["caption"]:
#     print(chunk, end="", flush=True)