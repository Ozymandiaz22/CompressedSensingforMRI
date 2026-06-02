import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import os

##generate a 150x150 bitmap thats all white, to be used as a test image for the reconstruction script
test_image = np.ones((150, 150)) * 255
plt.imsave("test_image.bmp", test_image, cmap='gray', vmin=0, vmax=255)
