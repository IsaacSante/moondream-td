from PIL import Image
import cv2
import time
import requests
from io import BytesIO
import base64

REMOTE_SERVER = "http://10.0.30.81:5001/infer"

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

try:
    while True:
        start = time.perf_counter()
        ok, frame_bgr = cap.read()
        if not ok:
            continue

        # Send full image with no compression
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        
        # Convert to base64 for API
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        image_data_uri = f"data:image/png;base64,{image_base64}"

        try:
            response = requests.post(REMOTE_SERVER, json={"image": image_data_uri}, timeout=5)
            result = response.json()
        except Exception as e:
            print(f"[ERROR] Failed to get inference result: {e}")
            result = {"objects": {}}

        # Display image with points
        display_frame = frame_bgr.copy()
        
        if result.get("objects"):
            for obj_name, points in result["objects"].items():
                for point in points:
                    x = int(point["x"] * frame_bgr.shape[1])
                    y = int(point["y"] * frame_bgr.shape[0])
                    
                    cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(display_frame, obj_name, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Detection", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
