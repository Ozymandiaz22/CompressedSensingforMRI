import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
from scipy import stats
import pylops
import pylops.optimization.sparsity 
from PIL import Image
import pydicom as dicom
import os


num_slices = 32
recons = []
originals = []

#seeks to do analysis on the generated data from Volumetric Wholedata
for i in range(num_slices):
    file = open(f"Volumetric/Reconstructed_images_whole/reconstructed_{i}.bmp", "rb")
    reconstructed_image = Image.open(file)
    reconstructed_image = np.array(reconstructed_image)
    recons.append(reconstructed_image)
    file.close()
    file = open(f"Volumetric/Reconstructed_images_whole/original_{i}.bmp", "rb")
    original_image = Image.open(file)
    original_image = np.array(original_image)
    originals.append(original_image)
    file.close()

#for reconstructed and original image select a 25*25 catch in the bottom left corner of the image
recons_snr = []
originals_snr = []


for i in range(len(recons)):
    recons_snr.append(recons[i][-49:-1,0:49])
    originals_snr.append(originals[i][-49:-1,0:49])

rms_recons = [np.sqrt(np.sum(x ** 2)/x.size) for x in recons_snr]
rms_originals = [np.sqrt(np.sum(x ** 2)/x.size) for x in originals_snr]

for i in range(len(recons)):
    print(f"Recons RMS for slice {i}: {rms_recons[i]}")
    print(f"Original RMS for slice {i}: {rms_originals[i]}")

#plot a bar graph of the RMS values for the reconstructed and original images
plt.figure(figsize=(10, 5))
x = np.arange(len(recons))
width = 0.35
plt.bar(x - width/2, rms_recons, width, label='Reconstructed RMS')
plt.bar(x + width/2, rms_originals, width, label='Original RMS')
plt.xlabel('Slice Index')
plt.ylabel('RMS Value')
plt.title('RMS Values for Reconstructed and Original Images')
plt.xticks(x)
plt.legend()
plt.savefig("Volumetric/RMS_comparison.png")
plt.show()
