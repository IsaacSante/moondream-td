"""
Call your MoonDream pod the same way you just did with curl,
but from Python.  No job polling loop needed.
"""
import base64, pathlib, requests

POD_URL = "https://fcaf0k2xvr5lvf-8000.proxy.runpod.net/chat/completions"
IMG_PATH = "dragon-icon.png"                   # local image to describe

def to_data_uri(path: str) -> str:
    mime = "image/png" if path.lower().endswith("png") else "image/jpeg"
    b64  = base64.b64encode(pathlib.Path(path).read_bytes()).decode()
    return f"data:{mime};base64,{b64}"

payload = {
    "model": "vikhyatk/moondream2",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s happening here?"},
                {"type": "image_url", "image_url": to_data_uri(IMG_PATH)}
            ]
        }
    ]
}

resp = requests.post(POD_URL, json=payload, timeout=120)
resp.raise_for_status()               # will raise if we got a non-200
print(resp.json()["choices"][0]["message"]["content"])
