import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
from scipy import stats
import pylops
import pylops.optimization.sparsity 
from PIL import Image
import pydicom as dicom
import os

import Live_parameters
import radiaslsampling as rss

####Parameters to set:
#folderpath to the dicom images
folderpath = Live_parameters.folderpath
#Image size to resize to
target_size = Live_parameters.target_size
##3D wavelet transform parameters
Wavelet_3D = Live_parameters.Wavelet_3D
Wavelet_3D_level = Live_parameters.Wavelet_3D_level
Percent_sampled = Live_parameters.Percent_sampled
Iteration_number = Live_parameters.Iteration_number
Regularization_Parameter = Live_parameters.Regularization_parameter
Tolerance = Live_parameters.Tolerance
Output_File = Live_parameters.Output_file

maskfile = Live_parameters.maskfile

plt.close('all')
#this file is to do volumetric reconstruction with total variation regularization
#find path of all dicom files in the folder
files = [str(folderpath + "/" + f  ) for f in os.listdir(folderpath) if f.endswith('.MRD')]
##previous dicom implementation
# dicoms = [dicom.dcmread(x) for x in files]
# images = [d.pixel_array for d in dicoms]

##read mrd file using evom3dread_converted
import evom3dread_converted as evom3d
mrd = evom3d.mread(files[0])
data = mrd['data']

#extract k spce images from the data
kspace = data[:, :, :, 0, 0, 0]
#this should be 150x150x150 for all real data sets

# Resize entire k-space to the next power of two for each dimension (center-crop or zero-pad)
def next_pow2(n):
    return 1 << ((n - 1).bit_length())

# original kspace shape: (nl0, ny0, nx0)
nl0, ny0, nx0 = kspace.shape
nl = next_pow2(nl0)
ny = next_pow2(ny0)
nx = next_pow2(nx0)

# If target_size is provided, override spatial dims to target_size
if 'target_size' in globals() and target_size is not None:
    ny = nx = target_size

# create new kspace array and center the original data
new_kspace = np.zeros((nl, ny, nx), dtype=kspace.dtype)
off_l = (nl - nl0) // 2
off_y = (ny - ny0) // 2
off_x = (nx - nx0) // 2

# compute source and destination slices
src_l0 = max(0, -off_l)
src_l1 = src_l0 + min(nl0, nl)
dst_l0 = max(0, off_l)

src_y0 = max(0, -off_y)
src_y1 = src_y0 + min(ny0, ny)
dst_y0 = max(0, off_y)

src_x0 = max(0, -off_x)
src_x1 = src_x0 + min(nx0, nx)
dst_x0 = max(0, off_x)

new_kspace[dst_l0:dst_l0+(src_l1-src_l0), dst_y0:dst_y0+(src_y1-src_y0), dst_x0:dst_x0+(src_x1-src_x0)] = \
    kspace[src_l0:src_l1, src_y0:src_y1, src_x0:src_x1]

# replace kspace with resized version and set dimensions
kspace = new_kspace
print("kspace shape after resizing:", kspace.shape)
#find the next power of 2 from the length of the array
#find the next power of two



Fop = pylops.signalprocessing.FFT2D(dims=(ny, nx))
Fop_3D = pylops.signalprocessing.FFTND(dims=(nl, ny, nx),ifftshift_before=True, dtype=np.complex128,fftshift_after=True)
Wop = pylops.signalprocessing.DWT(dims=ny*nx, wavelet='db6', level=1)
Wop3D = pylops.signalprocessing.DWTND(dims=(nl, ny, nx), wavelet=Wavelet_3D, level=Wavelet_3D_level, axes=(0, 1, 2), dtype=np.complex128)
Fop_3D_ifft = pylops.signalprocessing.FFTND(dims=(nl, ny, nx),ifftshift_before=True, dtype=np.complex128)


#set up the basis pursuit problem to be solved with FISTA algorithm assuming the image is sparse in wavelet domain
#our image comes to us undersampled in the fourier domain, so we need to use the fourier operator as our forward model
#we can use the wavelet transform as our sparsifying transform, and then use the inverse wavelet transform to reconstruct the image from the wavelet coefficients

sampling_regime = "1d"
if sampling_regime == "1d":
    perc_subsampling =  Percent_sampled
    line_length = nx
    nlinesub = int(np.round(line_length * perc_subsampling))
    ps = stats.norm.pdf(np.arange(line_length),loc = line_length/2,scale = 10)
    ps = ps/sum(ps)
    linesample = np.random.choice(line_length,size = nlinesub,p = ps,replace=False)
    linesample.sort()
    samples= []
    for i in range(ny*nl):
        samples.append([int(x) + line_length*i for x in linesample])
    samples = np.asarray(samples, dtype=np.int32)
    samples = samples.flatten()
    samples = list(samples)
    #restriction operator selects samples not to take, so we need to take the complement of the samples we want to take
    print(len(samples))
elif sampling_regime == "2d":
    ##there is an nx by ny grid, you are sampling a percentage of distinct points on this grid, and you are sampling the same points for each slice in the stack
    ##this should be done as a 2d gaussian sampling pattern
    perc_subsampling =  Percent_sampled
    n_sub = int(np.round(ny * nx * perc_subsampling))
    ps_y = stats.norm.pdf(np.arange(ny),loc = ny/2,scale = 30)
    ps_x = stats.norm.pdf(np.arange(nx),loc = nx/2,scale = 30)
    ps_y = ps_y/sum(ps_y)
    ps_x = ps_x/sum(ps_x)
    samples_2d = np.random.choice(ny*nx,size = n_sub,p = np.outer(ps_y, ps_x).flatten(),replace=False)
    samples_2d.sort()
    samples = []
    for i in range(nl):
        samples.append([int(x) + ny*nx*i for x in samples_2d])
    samples = np.asarray(samples, dtype=np.int32)
    samples = samples.flatten()
    samples = list(samples)
    #restriction operator selects samples not to take, so we need to take the complement of the samples we want to take
    #samples = list(all_samples - set(samples))
    print(len(samples))




#samples selected

#undersampled k_space generated

##print the sampling mask as an image
# Initialize mask with zeros (no samples) and set 1 where samples are taken
sampling_mask = np.zeros((nl, ny, nx), dtype=np.float32)
for s in samples:
    sampling_mask[s // (ny * nx), (s % (ny * nx)) // nx, s % nx] = 1

# show the first slice of the sampling mask (samples=1, missing=0)
plt.imshow(sampling_mask[0], cmap='gray', origin='lower')
plt.title('Sampling Mask for First Slice')
plt.imsave(f"Volumetric/sampling_mask.png", sampling_mask[0], cmap='gray')
plt.close()

#restriction obertor selects samples from the fourier domain
Rop = pylops.Restriction( nl *ny * nx, samples, axis=-1, dtype=np.complex128)
#our sparcifying tarnsform is the 3D wavelet transform
Sop = Wop3D

#we will seek to solve the analysis problem: given as
#argmin||y - Op x||_2^2 +epsilon*||SOp^H x||_1
#Sop is the wavelet transform, and Sop^H is the inverse wavelet transform
#Op is the forward model, which is the restriction operator composed with the fourier operator
#our undersampled k space is our measurements y, and our variable x is the image we want to reconstruct

Op = Rop * Fop_3D
#forward operator generated
y = Rop * np.asarray(kspace).ravel('K')
#measuremnents in the fourier domain generated in (nl*nx*samples,) shape
print("y shape:", y.shape)
#we can now use the FISTA algorithm to solve the optimization problem
epsilon = Regularization_Parameter
x0 = np.zeros((nl ,ny, nx), dtype=np.complex128).ravel('K')
#we will solve the problem for each image in the stack
recons = []
#print shapes of all the variables
print("Op shape:", Op.shape)
print("Sop shape:", Sop.shape)
print("x0 shape:", x0.shape)

#images = kspace * Fop_3D.H * (1/(nx*ny*nl))
images = Fop_3D_ifft.H * kspace.ravel('K')
images = images.reshape((nl, ny, nx))
#images = np.swapaxes(images, 0, 1)

# for i in range(len(y)):
#     print("reconstructing image", i)
#     (x, niter, cost) = pylops.optimization.sparsity.fista(Op, y[i], eps=epsilon, x0=x0, niter=100, SOp=Sop, tol=1e-6)
#     recons.append(x.reshape((ny, nx)))

#reconstruct the whole stack at once
(x, niter, cost) = pylops.optimization.sparsity.fista(Op, y, eps=epsilon, x0=x0, niter=Iteration_number, SOp=Sop, tol=Tolerance,show=True)
#un ravel the reconstructed stack
recons = x.reshape((nl, ny, nx))
print("reconstruction complete")
print("reconstructed image shape:", recons[0].shape)

##generate original images from the k space data by applying the inverse fourier transform to each slice in the stack
##3d ifft


print((images.shape))
print("original image dtype:", images.dtype)
images = np.asarray(images)
images = images.astype(np.complex128)
kspace = np.asarray(kspace)
kspace = kspace.astype(np.complex128) 
print("original image dtype:", images.dtype)
print("original image shape:", images[0].shape) 
print(images[0])
##convert scaled linear operator to a numpy array
print("reconslength:", len(recons))

difference_images = []
normalised_difference_images = []
for i in range(len(recons)):
    difference_image = np.abs(recons[i] - images[i])
    difference_images.append(difference_image)
    #for each recon and original image, normalise the difference image by dividing by peak value to scale to same magnitude
    peak_recon = np.max(np.abs(recons[i]))
    peak_original = np.max(np.abs(images[i]))
    if peak_recon == 0:
        peak_recon = 1
    if peak_original == 0:
        peak_original = 1
    normalised_difference_image = ((np.abs(recons[i])/peak_recon) - (np.abs(images[i])/peak_original))
    normalised_difference_images.append(normalised_difference_image)

####masked k spaces
masked_kspace = kspace * (sampling_mask)

#save a comparison of the original and reconstructed image for each slice in the stack to a folder
output_folder = Output_File
##Output file is now assumed to exist as it is being passed as an argument
for i in range(len(recons)):
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs[0, 0].imshow(np.abs(images[i]), cmap='gray')
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')
    axs[0, 1].imshow(np.abs(recons[i]), cmap='gray')
    axs[0, 1].set_title('Reconstructed Image')
    axs[0, 1].axis('off')
    axs[0, 2].imshow(np.abs(normalised_difference_images[i]), cmap='gray')
    axs[0, 2].set_title('Difference Image')
    axs[0, 2].axis('off')
    axs[1, 0].imshow(np.log(np.abs(kspace[i]) + 1e-10), cmap='gray')
    axs[1, 0].set_title('K-Space')
    axs[1, 0].axis('off')
    axs[1, 1].imshow(np.abs(sampling_mask[i]), cmap='gray')
    axs[1, 1].set_title('Sampling Mask')
    axs[1, 1].axis('off')
    axs[1, 2].imshow(np.log(np.abs(masked_kspace[i]) + 1e-10), cmap='gray')
    axs[1, 2].set_title('Masked K-Space')
    axs[1, 2].axis('off')
    plt.savefig(f"{output_folder}/comparison_{i}.png")
    plt.close(fig)
    plt.imsave(f"{output_folder}/reconstructed_{i}.bmp", np.abs(recons[i]), cmap='gray')
    plt.imsave(f"{output_folder}/original_{i}.bmp", np.abs(images[i]), cmap='gray')
    print(f"Saved comparison image for slice {i} to {output_folder}/comparison_{i}.png")
