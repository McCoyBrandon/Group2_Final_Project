# Group2_Final_Project

## I Spy Game

This portion of the project is a version of the game I Spy. The user plays against the computer, which acts as the guesser.

### Running the Game

The game is made using Gradio. Install any necessary prerequisite libraries, then run the code to open a Gradio server. A version uploaded to HuggingFace Spaces can be found at [https://huggingface.co/spaces/leandrumartin/ai-i-spy](https://huggingface.co/spaces/leandrumartin/ai-i-spy).

### The Gameplay

The user chooses or uploads a photo and privately chooses an object in the photo. The user then writes a clue to the computer of the form "I spy with my little eye something that is ___." The computer then scans the photo for objects that match the clue and presents a guess to the user. The user can make the computer guess again if the guess was incorrect. If the computer guesses correctly, the computer wins, otherwise the user does.

### Tools/Models/APIs and Inner Workings

The game uses two models available on HuggingFace. One is [facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50), which scans the inputted image for objects and returns a tensor of each object's classification, location in the image (the coordinates of the corners of the box bounding the object), and confidence in the model's classification.

The bounding box coordinates are used to crop the image into new images containing only one detected object each. These cropped images are passed into the second model used, [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32). This CLIP model is first used to create an embedding of the textual clue that the user writes, then to create embeddings of each of the cropped images. A cosine similarity score is taken between each image and the clue to see how well each detected object matches the clue. The objects are then sorted according to these similarity scores, then presented to the user in order. The effect is that the computer's guesses are presented to the user in order from best to worst.
