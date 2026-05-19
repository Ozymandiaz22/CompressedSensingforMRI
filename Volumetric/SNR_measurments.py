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
recons = np.array(recons)
originals = np.array(originals)
#images might be of any number of different sizes, but most likely a power of two, so either 256*256, 512*512

#for reconstructed and original image select a 25*25 catch in the bottom left corner of the image
recons_noise = []
originals_noise = []


for i in range(len(recons)):
    recons_noise.append(recons[i][-49:-1,0:49])
    originals_noise.append(originals[i][-49:-1,0:49])

rms_recons = [np.sqrt(np.sum(x ** 2)/x.size) for x in recons_noise]
rms_originals = [np.sqrt(np.sum(x ** 2)/x.size) for x in originals_noise]

#generate the mask for the signal region as a circle in the centre of the image, with a radius of 25% of the biggest circle in a unit square

signal_mask = spiic.circle_in_centre(recons, percentradius=0.25)

recons_signal = recons * signal_mask
originals_signal = originals * signal_mask


#signal mask is a 1 and zero mask of the same shape as the entire 3d image,
mean_signal_recons = [np.sum(recons_signal[i]) / np.sum(signal_mask) for i in range(len(recons))]
mean_signal_originals = [np.sum(originals_signal[i]) / np.sum(signal_mask) for i in range(len(originals))]


# calculate SNR for each slice
snr_recons = [signal / noise if noise != 0 else np.inf for signal, noise in zip(mean_signal_recons, rms_recons)]
snr_originals = [signal / noise if noise != 0 else np.inf for signal, noise in zip(mean_signal_originals, rms_originals)]


### use ski to find psnr and ssim for each slice, put results in a list
psnr_recons = [ski.metrics.peak_signal_noise_ratio(originals[i], recons[i]) for i in range(len(recons))]
ssim_recons = [ski.metrics.structural_similarity(originals[i], recons[i]) for i in range(len(recons))]


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
input_name = os.path.basename(os.path.normpath(filepath))
plt.savefig(os.path.join(outputpath, f"{input_name}_RMS_comparison.png"))
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, snr_recons, width, label='Reconstructed SNR')
plt.bar(x + width/2, snr_originals, width, label='Original SNR')
plt.xlabel('Slice Index')
plt.ylabel('SNR Value')
plt.title('SNR Values for Reconstructed and Original Images')
plt.xticks(x)
plt.legend()
plt.savefig(os.path.join(outputpath, f"{input_name}_SNR_comparison.png"))
plt.close()

##plot a bar graph of the PSNR values for the reconstructed images
plt.figure(figsize=(10, 5))
plt.bar(x, psnr_recons, width, label='Reconstructed PSNR')
plt.xlabel('Slice Index')
plt.ylabel('PSNR Value')
plt.title('PSNR Values for Reconstructed Images')
plt.xticks(x)
plt.legend()
plt.savefig(os.path.join(outputpath, f"{input_name}_PSNR_comparison.png"))
plt.close()

##plot a bar graph of the SSIM values for the reconstructed images
plt.figure(figsize=(10, 5))
plt.bar(x, ssim_recons, width, label='Reconstructed SSIM')
plt.xlabel('Slice Index')
plt.ylabel('SSIM Value')
plt.title('SSIM Values for Reconstructed Images')
plt.xticks(x)
plt.legend()
plt.savefig(os.path.join(outputpath, f"{input_name}_SSIM_comparison.png"))
plt.close()

#write a text file with the SNR, PSNR and SSIM values for each slice
with open(os.path.join(outputpath, f"{input_name}_metrics.txt"), 'w') as f:
    f.write("Slice Index\tRMS Reconstructed\tRMS Original\tSNR Reconstructed\tSNR Original\tPSNR Reconstructed\tSSIM Reconstructed\n")
    for i in range(len(recons)):
        f.write(f"{i}\t{rms_recons[i]:.4f}\t{rms_originals[i]:.4f}\t{snr_recons[i]:.4f}\t{snr_originals[i]:.4f}\t{psnr_recons[i]:.4f}\t{ssim_recons[i]:.4f}\n")
print(f"Metrics saved to {os.path.join(outputpath, f'{input_name}_metrics.txt')}")