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


filepath = "C:\\Users\\osman\\Documents\\FYP Datasets\\Batch 4 Results\\5927"

##get all the reg folders in the filepath
Regs_list = []
Regs_paths = []
for f in os.listdir(filepath):
    if f.startswith('Reg'):
        Regs_list.append(f.split(' ')[1])
        Regs_paths.append(os.path.join(filepath, f))

origs_whole = []
recons_whole = []

for Reg_path in Regs_paths:
    ##get all the bitmap files
    originals = []
    reconstructions = []
    print("Reg path: ", Reg_path)
    for f in os.listdir(Reg_path):
        if f.endswith('.bmp'):
            if 'original' in f:
                originals.append(Image.open(os.path.join(Reg_path, f)).convert('L'))
            elif 'reconstructed' in f:
                reconstructions.append(Image.open(os.path.join(Reg_path, f)).convert('L'))
    ##make all the images numpy arrays
    originals = np.array(originals)
    reconstructions = np.array(reconstructions)
    origs_whole.append(originals)
    recons_whole.append(reconstructions)

##show the 100th slice of the original and reconstructed images for the first reg folder
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(origs_whole[0][100], cmap='gray')
plt.title('Original Image (Reg ' + Regs_list[0] + ')')
plt.subplot(1, 2, 2)
plt.imshow(recons_whole[0][100], cmap='gray')
plt.title('Reconstructed Image (Reg ' + Regs_list[0] + ')')
plt.tight_layout()
plt.show()


##find the mean ssim and psnr for each reg folder
ssim_means = []
psnr_means = []
for i in range(len(Regs_list)):
    ssim_list = []
    psnr_list = []
    for j in range(len(origs_whole[i])):
        ssim_list.append(ski.metrics.structural_similarity(origs_whole[i][j], recons_whole[i][j], data_range=255))
        psnr_list.append(ski.metrics.peak_signal_noise_ratio(origs_whole[i][j], recons_whole[i][j], data_range=255))
    print("mean ssim for Reg " + Regs_list[i] + ": " + str(np.mean(ssim_list)))
    print("mean psnr for Reg " + Regs_list[i] + ": " + str(np.mean(psnr_list)))

    ssim_means.append(np.mean(ssim_list))
    psnr_means.append(np.mean(psnr_list))

print("Regs_list: ", Regs_list)
print("SSIM means: ", ssim_means)
print("PSNR means: ", psnr_means)

#plot the ssim and psnr means against the reg values
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(Regs_list, ssim_means, marker='o')
plt.xscale('log')
plt.xlabel('Regularization Parameter')
plt.ylabel('Mean SSIM')
plt.title('Mean SSIM vs Regularization Parameter')
plt.subplot(1, 2, 2)
plt.plot(Regs_list, psnr_means, marker='o')
plt.xscale('log')
plt.xlabel('Regularization Parameter')
plt.ylabel('Mean PSNR')
plt.title('Mean PSNR vs Regularization Parameter')
plt.tight_layout()
plt.show()

    
    