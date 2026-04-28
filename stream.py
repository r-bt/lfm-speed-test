import json
import requests
import base64
import cv2
import time
from pathlib import Path
# from io import BytesIO
# from PIL import Image

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1 sync=false"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

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
    "video/x-raw, format=BGR ! appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
writer = None
output_path = Path("output.mp4")
prev_time = time.time()
current_fps = 30.0

if not cap.isOpened():
    print("Camera failed to open")
    exit()

try:
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

        # Calculate FPS for the loop timing
        current_time = time.time()
        loop_fps = 1 / (current_time - prev_time)
        prev_time = current_time

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, current_fps, (new_w, new_h))

        # Encode the image to base64
        _, buffer = cv2.imencode('.jpg', frame)
        image_b64 = base64.b64encode(buffer).decode()

        payload = generate_payload("glasses", image_b64)
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

        writer.write(frame)

        # Display FPS on the live preview
        cv2.putText(frame, f"FPS: {loop_fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()