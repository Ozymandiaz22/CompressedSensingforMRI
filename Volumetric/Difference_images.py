import sys

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
filepath = sys.argv[1]
outputpath = sys.argv[2]

#seeks to do analysis on the generated data from Volumetric Wholedata
recons = []
originals = []

# Get all files in the directory
files = sorted(os.listdir(filepath))

# Parse files to find reconstructed and original bitmaps
for filename in files:
    if filename.endswith('.bmp'):
        full_path = os.path.join(filepath, filename)
        if 'reconstructed_' in filename:
            file = open(full_path, "rb")
            reconstructed_image = Image.open(file).convert('L')
            reconstructed_image = np.array(reconstructed_image)
            recons.append(reconstructed_image)
            file.close()
        elif 'original_' in filename:
            file = open(full_path, "rb")
            original_image = Image.open(file).convert('L')
            original_image = np.array(original_image)
            originals.append(original_image)
            file.close()



#genrate the difference images
difference_images = []
normalised_difference_images = []
for i in range(len(recons)):
    difference_image = np.abs(recons[i] - originals[i])
    difference_images.append(difference_image)
    #for each recon and original image, normalise the difference image by the mean of each image
    mean_recon = np.mean(recons[i])
    mean_original = np.mean(originals[i])
    if mean_recon == 0:
        mean_recon = 1
    if mean_original == 0:
        mean_original = 1
    normalised_difference_image = np.abs((recons[i]/mean_recon) - (originals[i]/mean_original))
    normalised_difference_images.append(normalised_difference_image)

#save the difference images to a folder
if not os.path.exists(outputpath):
    os.makedirs(outputpath)
for i in range(len(difference_images)):
    plt.imsave(f"{outputpath}/difference_{i}.bmp", difference_images[i], cmap='gray')
for i in range(len(normalised_difference_images)):
    plt.imsave(f"{outputpath}/normalised_difference_{i}.bmp", normalised_difference_images[i], cmap='gray')
