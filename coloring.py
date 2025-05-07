from PIL import Image
import numpy as np
import cv2


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