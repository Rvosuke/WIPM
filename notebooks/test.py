import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
image_path = "./transfer.png"
image = Image.open(image_path)

# Display the original image
plt.imshow(image)
plt.axis("off")
plt.title("Original Image with Gray Background")
plt.show()

# Convert the image to OpenCV format and RGB to BGR
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Convert to grayscale to isolate background
gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

# Threshold to identify background (gray values are around 220~240)
_, mask = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)

# Create white background
white_bg = np.ones_like(image_cv, dtype=np.uint8) * 255

# Combine the image with white background
result = np.where(mask[:, :, None] == 255, white_bg, image_cv)

# Convert back to PIL Image for saving/viewing
result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
# result_image.show()
output_path = "./pro_transfer.png"
result_image.save(output_path)
