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
import nibabel as nib



##import 1 dataset to test the sizing of the k space and the mask
mrd_filepath = "C:\\Users\\osman\\Documents\\FYP Datasets\\Batch 1\\V07_MRD\\5684"

files = [str(mrd_filepath + "/" + f  ) for f in os.listdir(mrd_filepath) if f.endswith('.MRD')]

##read mrd file using evom3dread_converted
import evom3dread_converted as evom3d
mrd = evom3d.mread(files[0])
data = mrd['data']
print("data shape:", data.shape)   
#extract k spce images from the data
kspace = data[:, :, :, 0, 0, 0]
#this should be 150x150x150 for all real data sets

print("kspace shape:", kspace.shape)
nl, ny, nx = kspace.shape
print("nl:", nl, "ny:", ny, "nx:", nx)



Wavelet_3D = 'haar'
Wavelet_3D_level = 7


Fop = pylops.signalprocessing.FFT2D(dims=(ny, nx))
Fop_3D = pylops.signalprocessing.FFTND(dims=(nl, ny, nx),ifftshift_before=True, dtype=np.complex128,fftshift_after=True)
Wop = pylops.signalprocessing.DWT(dims=ny*nx, wavelet='db6', level=1)
Wop3D = pylops.signalprocessing.DWTND(dims=(256, 256, 256), wavelet=Wavelet_3D, level=Wavelet_3D_level, axes=(-3, -2, -1), dtype=np.complex128)
CosOp = pylops.signalprocessing.DCT(dims=(nl, ny, nx))
Fop_3D_ifft = pylops.signalprocessing.FFTND(dims=(nl, ny, nx),ifftshift_before=True, dtype=np.complex128)
padl = (256-150)/2
pad_lengths = ((int(padl), int(padl)), (int(padl), int(padl)), (int(padl), int(padl)))
Pad_Op = pylops.Pad(dims=(nl, ny, nx), pad=pad_lengths, dtype=np.complex128)

maskfile = 'C:\\Users\\osman\\Documents\\GitHub\\CompressedSensingforMRI\\output\\exLUT_radial_trajectories\\exLUT_radial_trajectory_y150_z150_pct59.7_a16.95_r1.bmp'
mask = Image.open(maskfile).convert('L')
mask_array = np.array(mask)
print("mask shape:", mask_array.shape)
sampling_mask = np.stack([mask_array] * nl, axis=0)
samples = np.where(sampling_mask.flatten() == 255)[0]
print("number of samples:", len(samples))



Rop = pylops.Restriction(nl*ny*nx, samples, dtype=np.complex128)
print("Pad_Op shape:", Pad_Op.shape)
print("Wop3D shape:", Wop3D.shape)
Sop =   Wop3D 
#Sop = CosOp

Op = Rop @ Fop_3D @ Pad_Op.H
x_0 = np.zeros(256*256*256, dtype=np.complex128)
y = Rop @ np.asarray(kspace.flatten(), dtype=np.complex128)
recons = []

epsilon = 1e-6
Iteration_number = 20
Tolerance = 1e-6
##print the shapes of the operators and the data
print("Op shape:", Op.shape)
print("y shape:", y.shape)
print("x_0 shape:", x_0.shape)
print("Sop shape:", Sop.shape)

##shoe slice 100 of the k space and slice 100 of the padded k space


(x, niter, cost) = pylops.optimization.sparsity.fista(Op, y, eps=epsilon, x0=x_0, niter=Iteration_number, SOp=Sop, tol=Tolerance,show=True)
recons = Pad_Op.H * x.reshape((256, 256, 256)) 
print("recons shape:", recons.shape)

#show the reconstreud image as an animation
import matplotlib.animation as animation
fig = plt.figure()
ims = []
for i in range(nl):
    im = plt.imshow(np.abs(recons[i, :, :]), animated=True)
    #write the slice number on the image
    plt.text(10, 10, f'Slice {i}', color='white', fontsize=12)
    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)
plt.show()