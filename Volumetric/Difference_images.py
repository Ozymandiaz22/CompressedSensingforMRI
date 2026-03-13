import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
from scipy import stats
import os
from PIL import Image
import bm3d

##This method imports bitmaps of the original and reconstructed images
##returns the difference between the two images as a list of numpy arrays

##attempt to import all bitmaps from an inport folder
##for mat should be original_{i}.bmp and reconstructed_{i}.bmp where i is the slice number
##keep going until no more bitmaps are found, and store the images in a list of numpy arrays

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

#genrate the difference images
difference_images = []
for i in range(len(recons)):
    difference_image = np.abs(recons[i] - originals[i])
    difference_images.append(difference_image)


#normalise the images to the range 0-1, currently the images are in the range 0-255
for i in range(len(difference_images)):
    difference_images[i] = difference_images[i]/255



#save the difference images to a folder
output_folder = "Volumetric/Difference_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for i in range(len(difference_images)):
    plt.imsave(f"{output_folder}/difference_{i}.bmp", difference_images[i], cmap='gray')
