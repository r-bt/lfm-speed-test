import json

import requests
import base64
import time
from io import BytesIO
from PIL import Image

def generate_payload(image_b64):
    return {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image?"},
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

images = [
    "test_imgs/test1.jpg",
    "test_imgs/test2.jpg",
    "test_imgs/test3.jpg"
]

# Ask for the model we are using for this test

model = input("Enter the model you want to test?: ")

# Encode and rescale all the images in the list
encoded_images = []

for image_path in images:
    with open(image_path, "rb") as f:
        image = Image.open(f)
        image = image.resize((512, 512))

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode()
        encoded_images.append(image_b64)

url = "http://localhost:8080/v1/chat/completions"

results = []

total_start_time = time.time()
for encoded_image in encoded_images:
    start_time = time.time()
    payload = generate_payload(encoded_image)
    response = requests.post(url, json=payload)

    results.append({
        "response": response.json(),
        "time": time.time() - start_time,
        "predicted_per_second": response.json()['timings']['predicted_per_second']
    })

for idx, result in enumerate(results):
    print(f"Image {idx + 1}:")
    print(f"Response: {result['response']}")
    print(f"Time taken: {result['time']} seconds\n")

print("Total time taken for all images: {:.2f} seconds".format(time.time() - total_start_time))
print("Average time per image: {:.2f} seconds".format((time.time() - total_start_time) / len(encoded_images)))

with open("results.json", "w") as f:
    avg_predicted_per_second = sum(result['predicted_per_second'] for result in results) / len(results)

    full_results = {
        "model": model,
        "results": results,
        "total_time": time.time() - total_start_time,
        "average_time": (time.time() - total_start_time) / len(encoded_images),
        "average_predicted_per_second": avg_predicted_per_second
    }

    f.write(json.dumps(full_results, indent=4))