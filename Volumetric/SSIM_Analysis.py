import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
from scipy import stats
import pylops
import pylops.optimization.sparsity 
from PIL import Image
import pydicom as dicom
import os
import methods.sphereinisocentre as spiic
import sys
import skimage as ski


##this script will take in a folder of folders of bitmaps, where there are bitmaps for the original,
##-- the reconstructed and the normalised difference images

#these set of folders will then be parsed, and grouped into arrays to do analysis on
#labels of the data come from the names of the folders, that have the results 

#for example:

#results folder
#-dataset 1
#--percent sampled 0.4
#---list of bitmaps to read

#the script will then put these in a dictionary, where the keys are the titles of the bitmap folders

if len(sys.argv) > 1:
    filepath = sys.argv[1]
    outputpath = sys.argv[2]
else:
    filepath = "C:\\Users\\osman\\Documents\\FYP Datasets\\Batch 1\\HV15_MRD\\5927 Results"
    outputpath = "C:\\Users\\osman\\Documents\\FYP Datasets\\Batch 1\\HV15_MRD\\5927 Analysis"

##open the results folder, and get the list of subfolders, which are the different percent sampled values
percent_sampled_folders = [f for f in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, f))]
print("Percent sampled folders found:", percent_sampled_folders)    
#for each percent sampled folder, get the list of bitmaps, and group them into arrays of original, reconstructed and normalised difference images
data = {}
for folder in percent_sampled_folders:
    folder_path = os.path.join(filepath, folder)
    bitmaps = [f for f in os.listdir(folder_path) if f.endswith('.bmp')]
    #print(f"Bitmaps found in {folder}: {bitmaps}")
    original_images = []
    reconstructed_images = []
    normalised_difference_images = []
    for bitmap in bitmaps:
        if 'original' in bitmap:
            original_images.append(os.path.join(folder_path, bitmap))
        elif 'reconstructed' in bitmap:
            reconstructed_images.append(os.path.join(folder_path, bitmap))
        elif 'normalized_difference' in bitmap:
            normalised_difference_images.append(os.path.join(folder_path, bitmap))
    data[folder] = {
        'original': original_images,
        'reconstructed': reconstructed_images,
        'normalised_difference': normalised_difference_images
    }

##an SSIM measurement is done for each pair of original and reconstructed images, and the results are plotted as a function of percent sampled, to see how the SSIM changes with different sampling regimes
ssim_results = {}
for folder in data:
    original_images = data[folder]['original']
    reconstructed_images = data[folder]['reconstructed']
    ssim_values = []
    for original, reconstructed in zip(original_images, reconstructed_images):
        original_image = Image.open(original).convert('L')
        reconstructed_image = Image.open(reconstructed).convert('L')
        ssim_value = ski.metrics.structural_similarity(np.array(original_image), np.array(reconstructed_image))
        ssim_values.append(ssim_value)
    ssim_results[folder] = ssim_values

#plot the results as a 3d bar chart, with the x axis as the slice number, the y axis as the percent sampled and the z axis as the ssim value
import re

def parse_percent(folder_name: str) -> float:
    # handle names like "0_45" or "percent 0_45" -> 0.45
    token = folder_name.split(' ')[-1]
    token = token.replace('_', '.')
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", token)
    return float(m.group()) if m else 0.0

# sort folders by numeric percent so bars appear in correct order
sorted_folders = sorted(data.keys(), key=parse_percent)
percent_sampled_values = [parse_percent(f) for f in sorted_folders]
slice_numbers = list(range(len(data[sorted_folders[0]]['original'])))
ssim_values = [ssim_results[f] for f in sorted_folders]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, p in enumerate(percent_sampled_values):
    # ensure p is float; matplotlib accepts float positions
    ax.bar(slice_numbers, ssim_values[i], zs=float(p), zdir='y', alpha=0.8)

ax.set_xlabel('Slice Number')
ax.set_ylabel('Percent Sampled')
ax.set_zlabel('SSIM Value')
plt.title('SSIM Values for Different Percent Sampled Values')
plt.savefig(os.path.join(outputpath, "SSIM_Analysis.png"))

#plot the average SSIM value for each percent sampled value as a line graph
average_ssim_values = [np.mean(ssim_results[f]) for f in sorted_folders]
plt.figure()
plt.plot(percent_sampled_values, average_ssim_values, marker='o')
plt.xlabel('Percent Sampled')
plt.ylabel('Average SSIM Value')
plt.title('Average SSIM Value for Different Percent Sampled Values')
plt.grid()
plt.savefig(os.path.join(outputpath, "Average_SSIM_Analysis.png"))

