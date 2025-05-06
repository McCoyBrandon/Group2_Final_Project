# ---- Install packages in CMD Console ----
#pip install openai gradio transformers diffusers accelerate scipy torch torchaudio soundfile datasets opencv-python

# ---- I spy game packages ----
import gradio as gr
from transformers import DetrImageProcessor, DetrForObjectDetection, CLIPProcessor, CLIPModel
import torch

# ---- Story-telling Packages ----
from openai import OpenAI
#import gradio as gr
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from datasets import load_dataset
import soundfile as sf
from diffusers import StableDiffusionPipeline
import tempfile
import numpy as np

# ---- Coloring Outline Packages ----
import cv2
from PIL import Image

##
# ---- Section 1: I Spy Game ----
##
# Load models
obj_detector_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
obj_detector_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


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
# Setting ChatGPT keys
client = OpenAI(api_key="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")

# Load processor and model for voice narration
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

# Load speaker embeddings (use pretrained speaker from dataset)
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

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

# ---- Option 1: Narrate Story ----
def generate_audio(text):
    try:
        # Truncate to SpeechT5's limit
        text = text[:600]

        # Generate speech
        inputs = processor(text=text, return_tensors="pt")
        speech = model.generate_speech(inputs["input_ids"], speaker_embedding)

        # Convert to NumPy array, 1D float32
        waveform = speech.squeeze().cpu().numpy()
        waveform = np.clip(waveform, -1.0, 1.0)  # Ensure it's in [-1, 1]
        waveform = waveform.astype(np.float32)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, waveform, 16000, format="WAV", subtype="PCM_16")
            return tmp.name
    except Exception as e:
        print("Audio error:", e)
        return None


# ---- Optiion 2: Illustrate Story ----
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
# I Spy interface
with gr.Blocks() as i_spy_interface:
    gr.Markdown("## 🕵️ I Spy Game")
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload an image")
        clue_input = gr.Textbox(label="Give a clue", placeholder="e.g., something red")
    with gr.Row():
        guess_button = gr.Button("Start Game")
        next_guess_button = gr.Button("Guess again", visible=False)
    guess_text = gr.Textbox(label="Guess", interactive=False)
    guess_image = gr.Image(label="Guess Image", interactive=False)
    state = gr.State([0, [], [], []])  # initial state

    guess_button.click(start_game, inputs=[image_input, clue_input], outputs=[guess_text, guess_image, state, next_guess_button])
    next_guess_button.click(guess_again, inputs=state, outputs=[guess_text, guess_image])

# Storytelling interface
with gr.Blocks() as story_interface:
    gr.Markdown("## 📖 Storytelling Time")

    story_prompt = gr.Textbox(label="Story prompt", placeholder="e.g., A dragon that bakes cookies")

    # Output previews
    story_output = gr.Textbox(label="Generated Story", lines=10)
    story_button = gr.Button("Generate Story")
    audio_output = gr.Audio(label="Narration", type="filepath", interactive=True)
    audio_button = gr.Button("Generate Narration")
    image_output = gr.Gallery(label="Illustrations (SD or DALL·E)", columns=2, height="auto")
    image_button_sd = gr.Button("Generate with Stable Diffusion")
    image_button_dalle = gr.Button("Generate with ChatGPT: DALL·E")


    # Button logic
    story_button.click(generate_story, inputs=story_prompt, outputs=story_output)
    audio_button.click(generate_audio, inputs=story_output, outputs=audio_output)
    image_button_sd.click(generate_image, inputs=story_output, outputs=image_output)

    def display_dalle_images(story_text):
        urls = generate_dalle_images(story_text)
        return urls if urls else None

    image_button_dalle.click(display_dalle_images, inputs=story_output, outputs=image_output)

# Coloring Book interface
with gr.Blocks() as coloring_interface:
    gr.Markdown("## 🎨 Coloring Outline Generator")

    input_image = gr.Image(type="pil", label="Upload an Image")
    output_image = gr.Image(label="Coloring Outline")
    convert_button = gr.Button("Convert to Coloring Page")

    convert_button.click(convert_to_coloring_outline, inputs=input_image, outputs=output_image)

# Master interface
with gr.Blocks(title="Children Entertainment Assistant") as full_interface:
    gr.Markdown("# 🎉 Children Entertainment Assistant")
    with gr.Tab("I Spy Game"):
        i_spy_interface.render()
    with gr.Tab("Storytelling"):
        story_interface.render()
    with gr.Tab("Coloring Outlines"):
        coloring_interface.render()

# Launch the app
full_interface.launch()
