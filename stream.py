import json
import requests
import base64
import cv2
# import time
# from io import BytesIO
# from PIL import Image

def generate_payload(query: str, image_b64):
    prompt = f'Detect all instances of: {query}. Response must be a JSON array: [{{"label": ..., "bbox": [x1, y1, x2, y2]}}, ...]. Coordinates are normalized to [0,1].'

    return {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": 128,
    }

# Get live video stream with opencv

pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Camera failed to open")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        continue

    # Resize the frame so largest dim is 512 while keeping aspect ratio
    h, w, _ = frame.shape
    if h > w:
        new_h = 512
        new_w = int(w * (512 / h))
    else:
        new_w = 512
        new_h = int(h * (512 / w))

    frame = cv2.resize(frame, (new_w, new_h))

    # Encode the image to base64
    _, buffer = cv2.imencode('.jpg', frame)
    image_b64 = base64.b64encode(buffer).decode()

    payload = generate_payload("nose", image_b64)
    url = "http://localhost:8080/v1/chat/completions"
    response = requests.post(url, json=payload)

    det = json.loads(response.json()['choices'][0]['message']['content'])

    x1, y1, x2, y2 = det[0]['bbox']
    h, w, _ = frame.shape
    x1 = int(x1 * w)
    y1 = int(y1 * h)
    x2 = int(x2 * w)
    y2 = int(y2 * h)

    # Draw a bounding box around the detected person
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
