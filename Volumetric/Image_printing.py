import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

##fenerate a comparison figure of of the original, resonstructed, and difference images for a given slice number
slice_number = 17

original_image = Image.open(f"Volumetric/Reconstructed_images_whole/original_{slice_number}.bmp")
reconstructed_image = Image.open(f"Volumetric/Reconstructed_images_whole/reconstructed_{slice_number}.bmp")
difference_image = Image.open(f"Volumetric/Difference_images/difference_{slice_number}.bmp")

if not os.path.exists("Volumetric/Comparison_figures"):
    os.makedirs("Volumetric/Comparison_figures")

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(original_image, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[1].imshow(reconstructed_image, cmap='gray')
axs[1].set_title('Reconstructed Image')
axs[1].axis('off')
axs[2].imshow(difference_image, cmap='gray')
axs[2].set_title('Difference Image')
axs[2].axis('off')
plt.savefig(f"Volumetric/Comparison_figures/comparison_{slice_number}.png")
plt.show()
