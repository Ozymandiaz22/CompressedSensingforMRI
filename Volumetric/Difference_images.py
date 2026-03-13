import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
from scipy import stats
import os
from PIL import Image
import bm3d

##This method imports bitmaps of the original and reconstructed images
##returns the difference between the two images as a list of numpy arrays

##attempt to import all bitmaps from an import folder
##for mat should be original_{i}.bmp and reconstructed_{i}.bmp where i is the slice number
##keep going until no more bitmaps are found, and store the images in a list of numpy arrays
##import them as black and white images, so they are 2D arrays of pixel values

Folderpath = "Volumetric/Reconstructed_images_whole"
recons = []
originals = []
keepgoing = True

while keepgoing:
    i = len(recons)
    try:
        file = open(f"{Folderpath}/reconstructed_{i}.bmp", "rb")
        reconstructed_image = Image.open(file)
        reconstructed_image = np.array(reconstructed_image)
        recons.append(reconstructed_image)
        file.close()
        file = open(f"{Folderpath}/original_{i}.bmp", "rb")
        original_image = Image.open(file)
        original_image = np.array(original_image)
        originals.append(original_image)
        file.close()
    except FileNotFoundError:
        keepgoing = False


##turn images to grayscale if they are not already
for i in range(len(recons)):
    if len(recons[i].shape) == 3:
        recons[i] = np.mean(recons[i], axis=2)
    if len(originals[i].shape) == 3:
        originals[i] = np.mean(originals[i], axis=2)

#genrate the difference images
difference_images = []
normalised_difference_images = []
for i in range(len(recons)):
    difference_image = np.abs(recons[i] - originals[i])
    difference_images.append(difference_image)
    #for each recon and original image, normalise the difference image by the mean of each image
    mean_recon = np.mean(recons[i])
    mean_original = np.mean(originals[i])
    print(f"Mean of reconstructed image {i}: {mean_recon}")
    print(f"Mean of original image {i}: {mean_original}")
    if mean_recon == 0:
        mean_recon = 1
    if mean_original == 0:
        mean_original = 1
    normalised_difference_image = np.abs((recons[i]/mean_recon) - (originals[i]/mean_original))
    normalised_difference_images.append(normalised_difference_image)

#save the difference images to a folder
output_folder = "Volumetric/Difference_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for i in range(len(difference_images)):
    plt.imsave(f"{output_folder}/difference_{i}.bmp", difference_images[i], cmap='gray')
for i in range(len(normalised_difference_images)):
    plt.imsave(f"{output_folder}/normalised_difference_{i}.bmp", normalised_difference_images[i], cmap='gray')
