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
nx, ny = images[0].shape
print("image shape:", images[0].shape)

Fop = pylops.signalprocessing.FFT2D(dims=(ny, nx))
Wop = pylops.signalprocessing.DWT(dims=ny*nx, wavelet='db6', level=1)
Wop2D = pylops.signalprocessing.DWT2D(dims=(ny, nx), wavelet='db10', level=2)

kspace = [Fop * i for i in images]

#set up the basis pursuit problem to be solved with FISTA algorithm assuming the image is sparse in wavelet domain
#our image comes to us undersampled in the fourier domain, so we need to use the fourier operator as our forward model
#we can use the wavelet transform as our sparsifying transform, and then use the inverse wavelet transform to reconstruct the image from the wavelet coefficients

perc_subsampling = 0.20
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
print(len(samples))

#samples selected

mask = np.zeros((ny, nx), dtype=bool)
for i in range(ny):
    if i in linesample:
        mask[i, :] = True

#mask generated

undersampled_kspace = [k * mask for k in kspace]

#undersmaspled k_space genrated

Rop = pylops.Restriction(ny * nx, samples, axis=-1, dtype=np.complex128)
