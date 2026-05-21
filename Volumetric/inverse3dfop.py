import os
import numpy as np
import pylops
import matplotlib.pyplot as plt
import evom3dread_converted as evom3d

folderpath = "C:\\Users\\osman\\Documents\\FYP Datasets\\Batch 1\\HV15_MRD\\5927"
files = [str(folderpath + "/" + f  ) for f in os.listdir(folderpath) if f.endswith('.MRD')]
mrd = evom3d.mread(files[0])
data = mrd['data']
kspace = data[:, :, :, 0, 0, 0]


#reconstruct the images using the inverse 3D Fourier transform
nl, ny, nx = kspace.shape
print("kspace shape:", kspace.shape)
Fop_3D = pylops.signalprocessing.FFTND(dims=(nl, ny, nx),ifftshift_before=True)

images = Fop_3D.H * kspace.ravel('K') 
images = images.reshape((nl, ny, nx))
#swap axes 1 and 2 to get the correct orientation
images = np.swapaxes(images, 0, 1)
images = np.asarray(images)
images = images.astype(np.complex128)
print("images shape:", images.shape)
print("Absolute value of first image:", np.abs(images[100]))
plt.imshow(np.abs(images[100]), cmap='gray')
plt.title('Image')
plt.axis('off')
plt.show()
#show as animation

#chop off last 20 slices
images = images[:-20, :, :]

fig, ax = plt.subplots()
im = ax.imshow(np.abs(np.abs(images[0])), cmap='gray')
ax.set_title('Reconstructed Image')
ax.axis('off')

def animate(frame):
    im.set_array(np.abs(images[frame]))
    return [im]

from matplotlib.animation import FuncAnimation
anim = FuncAnimation(fig, animate, frames=images.shape[0], interval=100, blit=True)
plt.show()

#save as gif
anim.save('reconstructed_images.gif', writer='imagemagick', fps=5)
