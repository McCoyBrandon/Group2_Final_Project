# ---- Install packages in CMD Console ----
#pip install openai gradio transformers diffusers accelerate scipy torch torchaudio soundfile datasets opencv-python

# ---- I spy game packages ----
import gradio as gr
from transformers import DetrImageProcessor, DetrForObjectDetection, CLIPProcessor, CLIPModel
import torch
import os
from PIL import Image

# ---- Story-telling Packages ----
from openai import OpenAI
from diffusers import StableDiffusionPipeline
import numpy as np
import io
import base64

# ---- Coloring Outline Packages ----
import cv2

# ---- Setting ChatGPT keys ----
# If you have an OpenAI API account you can create an API key here: https://platform.openai.com/api-keys
client = OpenAI(api_key="YOUR_API_KEY_HERE")

##
# ---- Section 1: I Spy Game ----
##
# Load models
obj_detector_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
obj_detector_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function for selection images and preview during I Spy game
def get_selected_image(uploaded_img, example_name):
    if uploaded_img is not None:
        return uploaded_img
    if example_name:
        return Image.open(example_paths[example_name])
    return None

def update_preview(uploaded_img, example_name):
    if uploaded_img is not None:
        return uploaded_img
    if example_name:
        return Image.open(example_paths[example_name])
    return None

# Function starts the game interface
def start_game(image, clue):
    guess_state = [0, [], [], []]
    guess_number, similarity_calculations, text_guesses, image_guesses = guess_state

    # Detect objects in the image
    inputs = obj_detector_processor(images=image, return_tensors="pt")
    outputs = obj_detector_model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = obj_detector_processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.7
    )[0]
    zipped_results = list(zip(results["scores"], results["labels"], results["boxes"]))

    if not zipped_results:
        return "No objects detected with sufficient confidence.", None, guess_state, gr.update(visible=True)

    # Get embedding for clue
    processed_clue = clip_processor(text=clue, return_tensors="pt", padding=True)
    clue_embedding = clip_model.get_text_features(**processed_clue)
    clue_embedding = torch.nn.functional.normalize(clue_embedding, dim=-1)

    # Loop through detected object embeddings and calculate similarity to the clue embedding
    for score, label, box in zipped_results:
        box = [max(0, round(i)) for i in box.tolist()]  # Round and clip to non-negative
        box = [
            min(box[0], image.width),
            min(box[1], image.height),
            min(box[2], image.width),
            min(box[3], image.height)
        ]

        try:
            cropped_image = image.crop(box)
            label_text = obj_detector_model.config.id2label[label.item()]
            processed_image = clip_processor(images=cropped_image, return_tensors="pt")["pixel_values"]
            image_embeddings = clip_model.get_image_features(processed_image)
            image_embeddings = torch.nn.functional.normalize(image_embeddings, dim=-1)
            cosine_similarity = torch.nn.functional.cosine_similarity(clue_embedding, image_embeddings)

            similarity_calculations.append((cosine_similarity.item(), label_text, cropped_image))
        except Exception as e:
            print(f"Skipping object due to error: {e}")
            continue

    # Sort the embedding data by similarity to the clue embedding
    try:
        similarity_calculations.sort(key=lambda x: x[0], reverse=True)

        for similarity, label_text, cropped_image in similarity_calculations:
            text_guesses.append(label_text)
            image_guesses.append(cropped_image)

        return f'Is it this {text_guesses[0]}? If not, press "Guess again."', image_guesses[0], guess_state, gr.update(visible=True)
    except IndexError:
        return "I couldn't find anything in the image, so you win!", None, guess_state, gr.update(visible=True)

def guess_again(guess_state):
    guess_number, similarity_calculations, text_guesses, image_guesses = guess_state
    guess_number += 1
    try:
        guess_state[0] = guess_number  # Update guess number
        return f'Is it this {text_guesses[guess_number]}? If not, press "Guess again."', image_guesses[guess_number]
    except IndexError:
        return "I'm out of guesses. You win!", None
    
##
# ---- Section 2: Story-telling ----
##
saved_illustrations = [None] * 5  # Fixed 5 slots
SLOT_LABELS = [f"Image {i+1}" for i in range(5)]

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
        prompt_2 = f"{lines[-1]}. The image should not contain any text, labels, or words." if len(lines) > 1 else prompt_1
        #print("DALL·E prompt 1:", prompt_1) # Debugging purposes
        
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

##
# ---- Section 3: Coloring Outlines ----
##
# Setting ChatGPT keys
def convert_to_coloring_outline(input_image):
    try:
        image_np = np.array(input_image.convert("RGB"))
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)

        # Convert back to image
        output_image = Image.fromarray(sketch)
        return output_image
    except Exception as e:
        print("Coloring conversion error:", e)
        return None

##    
# ---- Final Gradio Interface ----
##
# Get example images
example_dir = "image_examples"
example_files = [f for f in os.listdir(example_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
example_paths = {os.path.splitext(f)[0]: os.path.join(example_dir, f) for f in example_files}

# I Spy interface
with gr.Blocks() as i_spy_interface:
    gr.Markdown("## 🕵️ I Spy Game")
    gr.Markdown("Upload or select an image, give a clue, and see if the AI can guess what you're thinking!")

    # Upload or select image
    with gr.Row():
        upload_input = gr.Image(type="pil", label="Upload your image")
        dropdown_examples = gr.Dropdown(
            label="Or choose a sample", 
            choices=list(example_paths.keys()), 
            interactive=True
        )

    # Preview selected image
    preview_image = gr.Image(type="pil", label="Selected Image", interactive=False)

    # Clue goes below preview, above the button
    clue_input = gr.Textbox(label="Give a clue: I spy with my little eye something that is...", placeholder="e.g., red")

    # Action buttons
    with gr.Row():
        guess_button = gr.Button("Start Game")
        next_guess_button = gr.Button("Guess again", visible=False)

    # Output: guess text and guess image
    guess_text = gr.Textbox(label="Guess", interactive=False)
    guess_image = gr.Image(label="Guess Image", interactive=False)
    state = gr.State([0, [], [], []])  # initial state

    # Image preview logic
    upload_input.change(fn=update_preview, inputs=[upload_input, dropdown_examples], outputs=preview_image)
    dropdown_examples.change(fn=update_preview, inputs=[upload_input, dropdown_examples], outputs=preview_image)

    # Start game logic
    guess_button.click(
        fn=lambda img, clue: start_game(img, clue),
        inputs=[preview_image, clue_input],
        outputs=[guess_text, guess_image, state, next_guess_button]
    )
    next_guess_button.click(guess_again, inputs=state, outputs=[guess_text, guess_image])

# Storytelling interface
with gr.Blocks() as story_interface:
    gr.Markdown("## 📖 Storytelling Time")
    gr.Markdown("Create a children's short story and choose from two illustration styles. Save your favorite images for coloring later!")

    story_prompt = gr.Textbox(label="Story prompt", placeholder="e.g., A dragon that bakes cookies")

    # Output previews
    story_output = gr.Textbox(label="Generated Story", lines=10)
    story_button = gr.Button("Generate Story")
    image_output = gr.Gallery(label="Illustrations (SD or DALL·E)", columns=2, height="auto")
    selected_image = gr.Image(type="pil", visible=False)
    image_button_sd = gr.Button("Generate with Stable Diffusion")
    image_button_dalle = gr.Button("Generate with ChatGPT: DALL·E")
    image_output.select(fn=return_selected_image, inputs=None, outputs=selected_image)
    save_slot_dropdown = gr.Dropdown(label="Save to Slot", choices=SLOT_LABELS, value=SLOT_LABELS[0])
    save_button = gr.Button("💾 Save to Coloring Book")


    def get_gallery_first(gallery):
        return gallery[0] if gallery else None

    save_button.click(
        fn=save_illustration,
        inputs=[selected_image, save_slot_dropdown],
        outputs=save_slot_dropdown
    )

    # Button logic
    story_button.click(generate_story, inputs=story_prompt, outputs=story_output)
    image_button_sd.click(generate_image, inputs=story_output, outputs=image_output)

    def display_dalle_images(story_text):
        urls = generate_dalle_images(story_text)
        return urls if urls else None

    image_button_dalle.click(display_dalle_images, inputs=story_output, outputs=image_output)

# Coloring Book interface
with gr.Blocks() as coloring_interface:
    gr.Markdown("## 🎨 Coloring Outline Generator")
    gr.Markdown("Upload your own image or select one you saved from the story section, then convert it into a fun black-and-white outline for coloring.")
    with gr.Row():
        uploaded_coloring = gr.Image(type="pil", label="Upload an Image")
        saved_dropdown = gr.Dropdown(label="Choose saved illustration", choices=SLOT_LABELS, interactive=True)

    coloring_preview = gr.Image(type="pil", label="Selected Image", interactive=False)

    uploaded_coloring.change(fn=lambda img: img, inputs=uploaded_coloring, outputs=coloring_preview)
    saved_dropdown.change(fn=get_saved_image, inputs=saved_dropdown, outputs=coloring_preview)
    
    output_image = gr.Image(label="Coloring Outline")
    convert_button = gr.Button("Convert to Coloring Page")

    convert_button.click(convert_to_coloring_outline, inputs=coloring_preview, outputs=output_image)

# Master interface
with gr.Blocks(title="Children Entertainment Assistant") as full_interface:
    gr.Markdown("# 🎉 Children Entertainment Assistant")
    gr.Markdown("Welcome! This assistant includes three fun and interactive tools for kids: an I Spy game, a storytelling prompt with illustrations, and a coloring page generator. "
    "Explore each tab to create, play, and learn!")
    with gr.Tab("I Spy Game"):
        i_spy_interface.render()
    with gr.Tab("Storytelling"):
        story_interface.render()
    with gr.Tab("Coloring Outlines"):
        coloring_interface.render()

# Launch the app
full_interface.launch()