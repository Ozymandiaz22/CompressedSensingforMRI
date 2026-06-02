import numpy as np
import matplotlib.pyplot as plt
##define a 64x64 grid of zeroes
grid_size = 20
grid = np.zeros((grid_size, grid_size))
##make 1 term at random location at amplitude 100
k_x = 2
k_y = 4
amplitude = 100
grid[k_x, k_y] = amplitude
##take the inverse Fourier transform to get the spatial pattern
spatial_pattern = np.fft.ifft2(grid)
##plot the spatial pattern as a 3D graph, with the aplitude on the z-axis
x = np.arange(grid_size)
y = np.arange(grid_size)
X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, spatial_pattern.real, cmap='viridis')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Amplitude')

plt.tight_layout()
plt.savefig(f'2D_Fourier_Term_kx_{k_x}_ky_{k_y}_amplitude_{amplitude}.png')

#plot the fouier space as a 3D graph as well, with the amplitude on the z-axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, grid, cmap='viridis')
ax.set_xlabel('k_x')
ax.set_ylabel('k_y')
ax.set_zlabel('Amplitude')
plt.tight_layout()
plt.savefig(f'2D_Fourier_Space_kx_{k_x}_ky_{k_y}_amplitude_{amplitude}.png')

