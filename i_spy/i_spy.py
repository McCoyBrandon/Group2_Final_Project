import gradio as gr
from transformers import DetrImageProcessor, DetrForObjectDetection, CLIPProcessor, CLIPModel
import torch

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
        return "No objects detected with sufficient confidence.", None, guess_state

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

        return f'Is it this {text_guesses[0]}? If not, press "Guess again."', image_guesses[0], guess_state
    except IndexError:
        return "I couldn't find anything in the image, so you win!", None, guess_state

def show_retry_button():
    return gr.update(visible=True)

def guess_again(guess_state):
    guess_number, similarity_calculations, text_guesses, image_guesses = guess_state
    guess_number += 1
    try:
        guess_state[0] = guess_number  # Update guess number
        return f'Is it this {text_guesses[guess_number]}? If not, press "Guess again."', image_guesses[guess_number]
    except IndexError:
        return "I'm out of guesses. You win!", None

with gr.Blocks() as demo:
    gr.Markdown("# 🕵️ I Spy Game with AI")

    guess_state = gr.State([0, [], [], []])

    with gr.Row():
        image_input = gr.Image(type="pil")
        clue_input = gr.Textbox(label="Enter your clue: I spy with my little eye something that is...", placeholder="I spy with my little eye something that is...")

    run_button = gr.Button("Start guessing")
    retry_button = gr.Button("Guess again", visible=False)

    output_text = gr.Markdown()
    output_image = gr.Image(label="Image Guess", interactive=False)

    run_button.click(fn=start_game, inputs=[image_input, clue_input], outputs=[output_text, output_image, guess_state])
    run_button.click(fn=show_retry_button, outputs=retry_button)

    retry_button.click(fn=guess_again, inputs=guess_state, outputs=[output_text, output_image])

    gr.Examples(
        examples=[
            "image_examples/busy_street.jpg",
            "image_examples/town_square.jpg",
            "image_examples/new_york_street.jpg",
        ],
        inputs=image_input
    )

    gr.Examples(
        examples=[
            "red",
            "green",
            "wearing clothes",
            "on wheels",
        ],
        inputs=clue_input
    )

demo.launch()
