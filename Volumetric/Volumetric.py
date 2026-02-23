import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
from scipy import stats
import pylops
import pylops.optimization.sparsity 
from PIL import Image
import pydicom as dicom
import os

#this file is to do volumetric reconstruction with total variation regularization
#find path of all dicom files in the folder
folderpath = "Volumetric/knee_mri_clinical_seq_batch2/1FB_1001820591____1FB,_3331562518/study_2f43b031/MR4_53c76c27"
files = [str(folderpath + "/" + f  ) for f in os.listdir(folderpath) if f.endswith('.dcm')]
dicoms = [dicom.dcmread(x) for x in files]
images = [d.pixel_array for d in dicoms]

#show images as a montage
##volumetric dicoms are doable
##volumetric reconstruction is attemtable

#we now have dicom images
#this scipt will use total variation regulariation in the wavelet domain to reconstruct the image
#image is sparse in wavelet domain, but initially aquiared int he fourier domain

#lets get fourier images using the fourier operator
#zero pad the images to all be 512x512
# Zero pad images to 512x512
target_size = 512
padded_images = []
for img in images:
    padded = np.zeros((target_size, target_size), dtype=img.dtype)
    h, w = img.shape
    padded[:h, :w] = img
    padded_images.append(padded)
images = padded_images

nx, ny = images[0].shape
print("image shape:", images[0].shape)

Fop = pylops.signalprocessing.FFT2D(dims=(ny, nx))
Wop = pylops.signalprocessing.DWT(dims=ny*nx, wavelet='db6', level=1)
Wop2D = pylops.signalprocessing.DWT2D(dims=(ny, nx), wavelet='db10', level=2)

kspace = [Fop * i for i in images]

#set up the basis pursuit problem to be solved with FISTA algorithm assuming the image is sparse in wavelet domain
#our image comes to us undersampled in the fourier domain, so we need to use the fourier operator as our forward model
#we can use the wavelet transform as our sparsifying transform, and then use the inverse wavelet transform to reconstruct the image from the wavelet coefficients

perc_subsampling = 0.50
line_length = nx
nlinesub = int(np.round(line_length * perc_subsampling))
ps = stats.norm.pdf(np.arange(line_length),loc = line_length/2,scale = 60)
ps = ps/sum(ps)
linesample = np.random.choice(line_length,size = nlinesub,p = ps,replace=False)
linesample.sort()
samples= []
for i in range(ny):
    samples.append([int(x) + line_length*i for x in linesample])
samples = np.asarray(samples, dtype=np.int32)
samples = samples.flatten()
samples = list(samples)
#restriction operator selects samples not to take, so we need to take the complement of the samples we want to take
all_samples = set(range(ny * nx))
samples = list(all_samples - set(samples))
print(len(samples))


#samples selected

mask = np.zeros((ny, nx), dtype=bool)
for i in range(ny):
    if i in linesample:
        mask[i, :] = True

#mask generated

undersampled_kspace = [k * mask for k in kspace]

#undersmaspled k_space genrated

#restriction obertor selects samples from the fourier domain
Rop = pylops.Restriction(ny * nx, samples, axis=-1, dtype=np.complex128)
#our sparcifying tarnsform is the 2D wavelet transform
Sop = Wop2D

#we will seek to solve the analysis problem: given as
#argmin||y - Op x||_2^2 +epsilon*||SOp^H x||_1
#Sop is the wavelet transform, and Sop^H is the inverse wavelet transform
#Op is the forward model, which is the restriction operator composed with the fourier operator
#our undersampled k space is our measurements y, and our variable x is the image we want to reconstruct

Op = Rop * Fop
#forward operator generated
y = [Rop * k.ravel() for k in kspace]
#measuremnents in the fourier domain generated in (nx*samples,) shape
print("y shape:", y[0].shape)

#we can now use the FISTA algorithm to solve the optimization problem
epsilon = 0.01
x0 = np.zeros((ny, nx), dtype=np.complex128).ravel()
#we will solve the problem for each image in the stack
recons = []
#print shapes of all the variables
print("Op shape:", Op.shape)
print("Sop shape:", Sop.shape)
print("x0 shape:", x0.shape)

for i in range(len(y)):
    print("reconstructing image", i)
    (x, niter, cost) = pylops.optimization.sparsity.fista(Op, y[i], eps=epsilon, x0=x0, niter=100, SOp=Sop, tol=1e-6)
    recons.append(x.reshape((ny, nx)))

#reconstruction complete, now we can show the images
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(images[5], cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[1].imshow(np.abs(recons[5]), cmap='gray')
axs[1].set_title('Reconstructed Image')
axs[1].axis('off')

#save a comparison of the original and reconstructed image for each slice in the stack to a folder
output_folder = "Volumetric/reconstructed_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for i in range(len(recons)):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(images[i], cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(np.abs(recons[i]), cmap='gray')
    axs[1].set_title('Reconstructed Image')
    axs[1].axis('off')
    plt.savefig(f"{output_folder}/comparison_{i}.png")
plt.show()
