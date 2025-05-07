import gradio as gr
import torch
from PIL import Image
from openai import OpenAI
from diffusers import StableDiffusionPipeline
import io
import base64

saved_illustrations = [None] * 5  # Fixed 5 slots
SLOT_LABELS = [f"Image {i + 1}" for i in range(5)]

# ---- Setting ChatGPT keys ----
# If you have an OpenAI API account you can create an API key here: https://platform.openai.com/api-keys
client = OpenAI(api_key="YOUR_API_KEY_HERE")

# Load Stable Diffusion model for image generation
device = "cuda" if torch.cuda.is_available() else "cpu"
sd_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", safety_checker=None
).to(device)


# ---- Step 1: Generate Story ----
def generate_story(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Write a short, whimsical children's story about: {prompt}"}
            ],
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Story error:", e)
        return f"Error generating story: {e}"


# ---- 2 Optiions for Illustrate Story ----
def generate_image(story_text):
    try:
        image = sd_pipe(story_text[:200]).images[0]
        return [image]
    except Exception as e:
        print("Image error:", e)
        return None


def generate_dalle_images(story_text):
    try:
        lines = [line.strip() for line in story_text.strip().split('\n') if line.strip()]
        if not lines:
            return None
        prompt_1 = f"{lines[0]}. The image should not contain any text, labels, or words."
        prompt_2 = f"{lines[-1]}. The image should not contain any text, labels, or words." if len(
            lines) > 1 else prompt_1
        # print("DALL·E prompt 1:", prompt_1) # Debugging purposes

        dalle_images = []
        for prompt in [prompt_1, prompt_2]:
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
            image_url = response.data[0].url
            dalle_images.append(image_url)
        return dalle_images
    except Exception as e:
        print("DALL·E error:", e)
        return None


def save_illustration(img, slot_label):
    try:
        # If the image is a base64 string, decode it
        if isinstance(img, str) and img.startswith("data:image"):
            header, base64_data = img.split(",", 1)
            img_bytes = base64.b64decode(base64_data)
            img = Image.open(io.BytesIO(img_bytes))

        index = SLOT_LABELS.index(slot_label)
        saved_illustrations[index] = img
        print(f"Image saved to slot {slot_label}")
        return gr.update(choices=SLOT_LABELS, value=slot_label)
    except Exception as e:
        print("Save failed:", e)
        return gr.update(choices=SLOT_LABELS)


def save_gallery_image(gallery):
    if gallery and len(gallery) > 0:
        img_data = gallery[0]
        if isinstance(img_data, tuple):
            img_path = img_data[0]
        else:
            img_path = img_data
        try:
            img = Image.open(img_path)
            return save_illustration(img)
        except Exception as e:
            print("Failed to open saved image:", e)
            return []
    return []


def extract_and_save(gallery, slot_label):
    if gallery and len(gallery) > 0:
        img_data = gallery[0]
        img_path = img_data[0] if isinstance(img_data, tuple) else img_data
        try:
            img = Image.open(img_path)
            return save_illustration(img, slot_label)
        except Exception as e:
            print("Failed to open image for saving:", e)
    return gr.update(choices=SLOT_LABELS)


import requests


def return_selected_image(evt: gr.SelectData):
    try:
        img_data = evt.value

        # Case 1: DALL·E dict with 'image' key and 'path'
        if isinstance(img_data, dict):
            if "image" in img_data and "path" in img_data["image"]:
                response = requests.get(img_data["image"]["path"])
                img = Image.open(io.BytesIO(response.content)).convert("RGB")
                return img

            # Case 2: data:image/... base64 in 'data' field
            if "data" in img_data and img_data["data"].startswith("data:image"):
                header, base64_data = img_data["data"].split(",", 1)
                img_bytes = base64.b64decode(base64_data)
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Case 3: base64 string directly
        if isinstance(img_data, str) and img_data.startswith("data:image"):
            header, base64_data = img_data.split(",", 1)
            img_bytes = base64.b64decode(base64_data)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Case 4: Already a PIL image
        if isinstance(img_data, Image.Image):
            return img_data

        print("[WARN] Unrecognized image format:", type(img_data), img_data)
        return None

    except Exception as e:
        print("Selection error:", e)
        return None


def get_saved_image(slot_label):
    try:
        index = SLOT_LABELS.index(slot_label)
        if saved_illustrations[index] is None:
            print(f"[INFO] Slot {slot_label} is empty.")
        else:
            print(f"[INFO] Loaded image from {slot_label}")
        return saved_illustrations[index]
    except Exception as e:
        print("Slot error:", e)
        return None
