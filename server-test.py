import requests
import base64
import time
from io import BytesIO

# Encode image
with open("test_image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Rescale the image to 512x512 using PIL
from PIL import Image
image = Image.open("test_image.jpg")
image = image.resize((512, 512))

# Convert the image to base64
buffered = BytesIO()
image.save(buffered, format="JPEG")
image_b64 = base64.b64encode(buffered.getvalue()).decode()


url = "http://localhost:8080/v1/chat/completions"

payload = {
    "model": "default",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
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

start_time = time.time()
response = requests.post(url, json=payload)
print(response.json())
print("Execution time: {:.2f} seconds".format(time.time() - start_time))