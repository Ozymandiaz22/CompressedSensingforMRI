import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import evom3dread_converted as evom3d
import PIL.Image as Image
import os

datapath = "C:/Users/osman/Documents/FYP Datasets/Batch 1/HV15_MRD/5927/5927_000_0.MRD"
mrd = evom3d.mread(datapath)
data = mrd['data']


#print the 1st slice of the experiment
slice1 = data[:, :, 0, 0, 0, 0]
plt.imshow(np.abs(slice1), cmap='gray')
print("Data shape:", data.shape)
print(slice1)
plt.show()