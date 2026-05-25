import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
from scipy import stats
import pylops
import pylops.optimization.sparsity 
from PIL import Image
import pydicom as dicom
import os



folderpath = "Volumetric/knee_mri_clinical_seq_batch2/1FB_1001820591____1FB,_3331562518/study_2f43b031/MR4_53c76c27"
files = [str(folderpath + "/" + f  ) for f in os.listdir(folderpath) if f.endswith('.dcm')]
dicoms = [dicom.dcmread(x) for x in files]
images = [d.pixel_array for d in dicoms]


##choose one image to test on
image_index = 10

##generate the k space data for the image
ny, nx = images[image_index].shape

Fop = pylops.signalprocessing.FFT2D(dims=(ny, nx),ifftshift_before=False, fftshift_after=True)
Wop = pylops.signalprocessing.DWT(dims=ny*nx, wavelet='db6', level=1)
Wop2D = pylops.signalprocessing.DWT2D(dims=(ny, nx), wavelet='db10', level=2)

k_space = Fop * images[image_index]

##select an inner portion of the k space to show the diference in encoded data
k_space_cropped = np.zeros_like(k_space)
crop_size = 8
k_space_cropped[ny//2-crop_size:ny//2+crop_size, nx//2-crop_size:nx//2+crop_size] = k_space[ny//2-crop_size:ny//2+crop_size, nx//2-crop_size:nx//2+crop_size]
##select the outer region of the k space to show the diference in encoded data
k_space_outer = np.copy(k_space)
k_space_outer[ny//2-crop_size:ny//2+crop_size, nx//2-crop_size:nx//2+crop_size] = 0
##2,3 figure with each k space and the image
inner_image = np.abs(Fop.H * k_space_cropped)
outer_image = np.abs(Fop.H * k_space_outer)
#show each k space and their respective images
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs[0, 0].imshow(np.log(np.abs(k_space)), cmap='gray')
axs[0, 0].set_title('Full k-space')
axs[0, 1].imshow(np.log(np.abs(k_space_cropped)), cmap='gray')
axs[0, 1].set_title('Cropped k-space')
axs[0, 2].imshow(np.log(np.abs(k_space_outer)), cmap='gray')
axs[0, 2].set_title('Outer k-space')
axs[1, 0].imshow(np.abs(Fop.H * k_space), cmap='gray')
axs[1, 0].set_title('Reconstructed from full k-space')
axs[1, 1].imshow(inner_image, cmap='gray')
axs[1, 1].set_title('Reconstructed from cropped k-space')
axs[1, 2].imshow(outer_image, cmap='gray')
axs[1, 2].set_title('Reconstructed from outer k-space')
#adjust layout to minimise whitespace
plt.tight_layout()

plt.show()

