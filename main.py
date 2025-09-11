import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology
from PIL import Image


#calculate POC
image = Image.open("sample.jpg")
image_array = np.array(image)


# Load image (replace 'sample.jpg' with your cane sample)
img = cv2.imread('sample.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur (reduce noise)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold using Otsu (automatic thresholding)
thresh_val = filters.threshold_otsu(blur)
binary = blur > thresh_val

# Remove small noise (morphology opening)
binary_clean = morphology.remove_small_objects(binary, 50)

# Show results
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(gray, cmap='gray'); ax[0].set_title("Gray")
ax[1].imshow(binary, cmap='gray'); ax[1].set_title("Binary")
ax[2].imshow(binary_clean, cmap='gray'); ax[2].set_title("Cleaned")
for a in ax: a.axis("off")
plt.show()

#calculate POC

# Load the sample image
image = Image.open("sample.jpg")

# Convert the image to a numpy array
image_array = np.array(image)

# Simple classification example: calculate average brightness
avg_brightness = image_array.mean()

if avg_brightness < 85:
    classification = "Dark"
elif avg_brightness < 170:
    classification = "Medium"
else:
    classification = "Bright"
    
poc_value = (avg_brightness / 255) * 100
poc_value = round(poc_value, 2)  # two decimals

print(f"POC result for sample.jpg: {poc_value}")

if poc_value >= 90:
    quality = "Good"
elif poc_value >= 80:
    quality = "Acceptable"
else:
    quality = "Poor"

print(f"POC result for sample.jpg: {poc_value} ({quality})")

