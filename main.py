from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from transformers.image_utils import load_image
import time

# Load model and processor
model_id = "LiquidAI/LFM2.5-VL-1.6B"
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto",
    dtype="bfloat16"
)
processor = AutoProcessor.from_pretrained(model_id)

# TIme how long inference takes
start_time = time.time()

# Load image and create conversation
url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
image = load_image(url)
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What is the capital of france"}
        ],
    },
]

# Generate Answer
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    tokenize=True,
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64)
res = processor.batch_decode(outputs, skip_special_tokens=True)[0]

# This image showcases the iconic Statue of Liberty standing majestically on Liberty Island in New York Harbor. The statue is positioned on a small island surrounded by calm blue waters, with the New York City skyline visible in the background.
print(res)

print("Execution time: {:.2f} seconds".format(time.time() - start_time))