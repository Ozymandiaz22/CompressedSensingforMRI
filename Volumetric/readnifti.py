##open nifti file and read the array

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os

nifti_img = nib.load("Volumetric/reconstructed_stack.nii")
##load the data as a numpy array
data = nifti_img.get_fdata()
print(data.shape)
##show the slices as an animation
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
im = ax.imshow(data[0, :, :], cmap='gray', animated=True)
ax.axis('off')

#scale the data to be between 0 and 255
data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
data = data.astype(np.uint8)


def update(i):
    im.set_array(data[i, :, :])
    ax.set_title(f"Slice {i}")
    return [im]

ani = FuncAnimation(fig, update, frames=data.shape[0], interval=100, blit=True, repeat=True)
plt.show()

#save salice 120 as a png image
plt.imsave("Volumetric/slice120.png", data[120, :, :], cmap='gray', format='png')
