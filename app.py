# ---- Install packages in CMD Console ----
#pip install openai gradio transformers diffusers accelerate scipy torch torchaudio soundfile datasets opencv-python

# ---- Import from local files ----
from i_spy import *
from storytelling import *
from coloring import *

##    
# ---- Final Gradio Interface ----
##
# Get example images

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