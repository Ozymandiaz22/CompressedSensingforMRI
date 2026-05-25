import sys
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
from scipy import stats
import pylops
import pylops.optimization.sparsity 
from PIL import Image
import pydicom as dicom
import os
import bm3d
import skimage as ski
import methods.sphereinisocentre as spiic

bitmaps_folderpath = 'C:\\Users\\osman\\Documents\\FYP Datasets\\Test Batch 3 Results\\5927 Results\\48_7\\'
##3 sets of bitmaps exist in the folder
#originals with'original' in the name
#reconstructions with 'reconstructed' in the name
#normalised differences with 'difference' in the name

##load the original, reconstructed and difference images
originals = []
reconstructions = []
differences = []
for f in os.listdir(bitmaps_folderpath):
    if f.endswith('.bmp'):
        if 'original' in f:
            originals.append(Image.open(os.path.join(bitmaps_folderpath, f)).convert('L'))
        elif 'reconstructed' in f:
            reconstructions.append(Image.open(os.path.join(bitmaps_folderpath, f)).convert('L'))
        elif 'difference' in f:
            differences.append(Image.open(os.path.join(bitmaps_folderpath, f)).convert('L'))


##denoise the reconstructions using bm3d
denoised_reconstructions = []
for recon in reconstructions:
    print("i = " + str(len(denoised_reconstructions)))
    recon_array = np.array(recon)
    denoised = bm3d.bm3d(recon_array, sigma_psd=25/255)
    denoised_reconstructions.append(denoised)

#make all the images numpy arrays
originals = np.array(originals)
reconstructions = np.array(reconstructions)
differences = np.array(differences)

##show slice 100 of the original, reconstructed, difference and denoised images
##get ssim and psnr and roi snr for the original and reconstructed images and the original and denoised reconstructions
###ssim
ssim_recon = []
ssim_denoised = []
for i in range(len(reconstructions)):
    ssim_recon.append(ski.metrics.structural_similarity(np.array(originals[i]), np.array(reconstructions[i]), data_range=255))
    ssim_denoised.append(ski.metrics.structural_similarity(np.array(originals[i]), denoised_reconstructions[i], data_range=255))
###psnr
psnr_recon = []
psnr_denoised = []
for i in range(len(reconstructions)):
    psnr_recon.append(ski.metrics.peak_signal_noise_ratio(np.array(originals[i]), np.array(reconstructions[i]), data_range=255))
    psnr_denoised.append(ski.metrics.peak_signal_noise_ratio(np.array(originals[i]), denoised_reconstructions[i], data_range=255))
###roi snr
roi_snr_recon = []
roi_snr_denoised = []
signal_mask = spiic.circle_in_centre(reconstructions, percentradius=0.25)
##noisemask is a 36x36 square in the bottom left corner of the image, starting from the edge of the image and going inwards, and starting from the edge of the image and going upwards, so it covers the area from (0,0) to (36,36) in the image coordinates
noisemask = np.zeros_like(signal_mask)
noisemask[-36:, :36] = 1
for i in range(len(reconstructions)):
    signal = np.mean(reconstructions[i] * signal_mask)
    noise = np.mean(reconstructions[i] * noisemask)
    roi_snr_recon.append(signal / noise)
    signal_denoised = np.mean(denoised_reconstructions[i] * signal_mask)
    noise_denoised = np.mean(denoised_reconstructions[i] * noisemask)
    roi_snr_denoised.append(signal_denoised / noise_denoised)



slice_index = 100
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
axs[0].imshow(originals[slice_index], cmap='gray')
axs[0].set_title('Original')
axs[1].imshow(reconstructions[slice_index], cmap='gray')
axs[1].set_title('Reconstructed')
axs[2].imshow(differences[slice_index], cmap='gray')
axs[2].set_title('Difference')
axs[3].imshow(denoised_reconstructions[slice_index], cmap='gray')
axs[3].set_title('Denoised Reconstruction')
plt.tight_layout()
plt.show()

##plot the ssim, psnr and roi snr values for the reconstructions and denoised reconstructions
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
x = np.arange(len(reconstructions))
width = 0.35

# SSIM difference plot
ssim_diff = np.array(ssim_denoised) - np.array(ssim_recon)
axs[0].bar(x, ssim_diff, width, label='Denoised - Reconstruction SSIM')
axs[0].set_xlabel('Slice Index')
axs[0].set_ylabel('SSIM Difference')
axs[0].set_title('SSIM Difference (Denoised - Reconstruction)')
axs[0].set_xticks(x)
axs[0].legend()

# PSNR difference plot
psnr_diff = np.array(psnr_denoised) - np.array(psnr_recon)
axs[1].bar(x, psnr_diff, width, label='Denoised - Reconstruction PSNR')
axs[1].set_xlabel('Slice Index')
axs[1].set_ylabel('PSNR Difference')
axs[1].set_title('PSNR Difference (Denoised - Reconstruction)')
axs[1].set_xticks(x)
axs[1].legend()

# ROI SNR difference plot
roi_snr_diff = np.array(roi_snr_denoised) - np.array(roi_snr_recon)
axs[2].bar(x, roi_snr_diff, width, label='Denoised - Reconstruction ROI SNR')
axs[2].set_xlabel('Slice Index')
axs[2].set_ylabel('ROI SNR Difference')
axs[2].set_title('ROI SNR Difference (Denoised - Reconstruction)')
axs[2].set_xticks(x)
axs[2].legend()

plt.tight_layout()
plt.show()
#save the charts
output_folder = 'C:\\Users\\osman\\Documents\\FYP Datasets\\Test Batch 3 Results\\5927 Analysis\\'
os.makedirs(output_folder, exist_ok=True)

# Save individual images from the first figure (slice views)
plt.imsave(os.path.join(output_folder, f"Original_slice_{slice_index}.png"), originals[slice_index], cmap='gray', vmin=0, vmax=255)
plt.imsave(os.path.join(output_folder, f"Reconstructed_slice_{slice_index}.png"), reconstructions[slice_index], cmap='gray', vmin=0, vmax=255)
plt.imsave(os.path.join(output_folder, f"Difference_slice_{slice_index}.png"), differences[slice_index], cmap='gray', vmin=0, vmax=255)
plt.imsave(os.path.join(output_folder, f"Denoised_slice_{slice_index}.png"), denoised_reconstructions[slice_index], cmap='gray', vmin=0, vmax=255)

# Save each subplot of the comparison bar charts as individual figures
# SSIM diff
fig_ssim, ax_ssim = plt.subplots(figsize=(6,4))
ax_ssim.bar(x, ssim_diff, width)
ax_ssim.set_xlabel('Slice Index')
ax_ssim.set_ylabel('SSIM Difference')
ax_ssim.set_title('SSIM Difference (Denoised - Reconstruction)')
fig_ssim.tight_layout()
fig_ssim.savefig(os.path.join(output_folder, "SSIM_comparison.png"))
plt.close(fig_ssim)

# PSNR diff
fig_psnr, ax_psnr = plt.subplots(figsize=(6,4))
ax_psnr.bar(x, psnr_diff, width)
ax_psnr.set_xlabel('Slice Index')
ax_psnr.set_ylabel('PSNR Difference')
ax_psnr.set_title('PSNR Difference (Denoised - Reconstruction)')
fig_psnr.tight_layout()
fig_psnr.savefig(os.path.join(output_folder, "PSNR_comparison.png"))
plt.close(fig_psnr)

# ROI SNR diff
fig_roi, ax_roi = plt.subplots(figsize=(6,4))
ax_roi.bar(x, roi_snr_diff, width)
ax_roi.set_xlabel('Slice Index')
ax_roi.set_ylabel('ROI SNR Difference')
ax_roi.set_title('ROI SNR Difference (Denoised - Reconstruction)')
fig_roi.tight_layout()
fig_roi.savefig(os.path.join(output_folder, "ROI_SNR_comparison.png"))
plt.close(fig_roi)


