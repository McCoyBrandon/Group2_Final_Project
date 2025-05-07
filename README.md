# Group2_Final_Project: Children Entertainment Assistant
---
## Introduction

The **Children Entertainment Assistant** is an interactive, AI-powered application designed to engage children through creativity, play, and storytelling. It combines multiple AI tools into a single, easy-to-use Gradio interface featuring:

- 🕵️ **I Spy Game**: An object recognition game powered by computer vision and CLIP embeddings where children provide a clue, and the AI tries to guess what they see in an image.
- 📖 **Storytelling**: A story generator that creates children's stories from a prompt, then narrates and illustrates them using language and image models.
- 🎨 **Coloring Outlines**: A tool that converts uploaded images into savable coloring book-style outlines using edge detection and image processing.

The app is built with Hugging Face Transformers, OpenAI’s ChatGPT and DALL·E APIs, and open-source models — all accessible through a single, friendly interface.

---
## Requirements

- Python 3.8 or higher
- [OpenAI API key](https://platform.openai.com/account/api-keys) for story generation and DALL·E image creation
- Compatible GPU (optional, but recommended for Stable Diffusion)
- The following Python libraries:
  - `gradio`
  - `openai`
  - `transformers`
  - `diffusers`
  - `accelerate`
  - `torch`, `torchaudio`
  - `soundfile`, `scipy`
  - `datasets`
  - `opencv-python`

You can install all required libraries using:

pip install -r requirements.txt

## How to run

1. Clone or download this repository to your local machine.

2. Open a terminal and navigate to the project directory.

3. (Optional) Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

4. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5. Open the `app.py` file and insert your OpenAI API key where indicated:

    ```python
    client = OpenAI(api_key = "YOUR_API_KEY_HERE")
    ```

6. Launch the app:

    ```bash
    python app.py
    ```

7. Your default browser will open a local Gradio interface with the following tabs:
    - 🕵️ I Spy Game
    - 📖 Storytelling
    - 🎨 Coloring Outlines

---

# Details about the app

## I Spy Game

This portion of the project is a version of the game I Spy. The user plays against the computer, which acts as the guesser.

### Running the Game

The game is made using Gradio. Install any necessary prerequisite libraries, then run the code to open a Gradio server. A version uploaded to HuggingFace Spaces can be found at [https://huggingface.co/spaces/leandrumartin/ai-i-spy](https://huggingface.co/spaces/leandrumartin/ai-i-spy).

### The Gameplay

The user chooses or uploads a photo and privately chooses an object in the photo. The user then writes a clue to the computer of the form "I spy with my little eye something that is ___." The computer then scans the photo for objects that match the clue and presents a guess to the user. The user can make the computer guess again if the guess was incorrect. If the computer guesses correctly, the computer wins, otherwise the user does.

### Tools/Models/APIs and Inner Workings

The game uses two models available on HuggingFace. One is [facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50), which scans the inputted image for objects and returns a tensor of each object's classification, location in the image (the coordinates of the corners of the box bounding the object), and confidence in the model's classification.

The bounding box coordinates are used to crop the image into new images containing only one detected object each. These cropped images are passed into the second model used, [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32). This CLIP model is first used to create an embedding of the textual clue that the user writes, then to create embeddings of each of the cropped images. A cosine similarity score is taken between each image and the clue to see how well each detected object matches the clue. The objects are then sorted according to these similarity scores, then presented to the user in order. The effect is that the computer's guesses are presented to the user in order from best to worst.

---

## Storytelling

This section allows the user to generate a whimsical children's story, narrate it with AI-generated speech, and illustrate it using AI-generated images.

- **Story Generator**: Uses OpenAI's ChatGPT API to generate a children's story based on a prompt.
- **Image Generation Options**:
  - **Stable Diffusion**: Creates a single image representing the first 600 characters of the story using the `runwayml/stable-diffusion-v1-5` model.
  - **DALL·E (ChatGPT)**: Generates two images using the first and last lines of the story by calling the OpenAI DALL·E 3 API.

The images are presented in the interface and can be downloaded.  Images can be included in the final section as well for coloring section.

---

## Coloring Outlines

This section transforms any uploaded image into a coloring book-style outline using OpenCV edge-preserving filtering. The user uploads a photo, and the interface returns a pencil-style sketch.

This is accomplished through the following image processing:

1. Convert the image to grayscale.
2. Invert the grayscale image.
3. Apply Gaussian blur.
4. Use division-based blending to simulate a pencil sketch effect.

This feature provides a fun way for users to convert real-world photos or generated images into coloring pages.
