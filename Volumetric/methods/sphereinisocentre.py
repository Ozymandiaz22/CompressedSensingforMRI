import numpy as np
import matplotlib.pyplot as plt

### this script defines a function to accept a 3d image and return a mask of a nomrlaised sphere in the isocentre of the image, as 1s and zeros.

def sphere_in_isocentre(image, percentradius):
    ##image is a 3d numpy array
    ##percentradius is the percent radius of the sphere in relation to the biggest sphere in a unit cube
    #get the dimensions of the image
    x_dim, y_dim, z_dim = image.shape
    #normalise the dimensions to be between 0 and 1
    x = np.linspace(0, 1, x_dim)
    y = np.linspace(0, 1, y_dim)
    z = np.linspace(0, 1, z_dim)
    #create a meshgrid of the normalised dimensions
    X, Y, Z = np.meshgrid(x, y, z)
    ##calculate the distance from the centre of the image for each point in the meshgrid
    distance_from_centre = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2)
    #calculate the radius of the sphere in relation to the biggest sphere in a unit cube
    radius = percentradius * (0.5) #the biggest sphere in a unit cube has a radius of 0.5
    #create a mask of the sphere
    mask = distance_from_centre <= radius
    #make the mask binary, with 1s for the sphere and 0s for the background
    mask = mask.astype(int)
    #mask has axis 0 moved to axis 1, and axis 1 moved to axis 0, so the shape is the same as the input image
    mask = np.transpose(mask, (1, 0, 2))
    return mask

def circle_in_centre(image,percentradius):
    ##image is a 3d numpy array, get a mask for each slice in the image set
    ##percentradius is the percent radius of the circle in relation to the biggest circle in a unit square
    #get the dimensions of the image
    x_dim, y_dim, z_dim = image.shape
    #normalise the dimensions to be between 0 and 1
    Y = np.linspace(0, 1, y_dim)
    Z = np.linspace(0, 1, z_dim)
    #create a meshgrid of the normalised dimensions
    Y, Z = np.meshgrid(Y, Z)
    ##calculate the distance from the centre of the image for each point in the meshgrid
    distance_from_centre = np.sqrt((Y - 0.5) ** 2 + (Z - 0.5) ** 2)
    #calculate the radius of the circle in relation to the biggest circle in a unit square
    radius = percentradius * (0.5) #the biggest circle in a unit square has a radius of 0.5
    #create a mask of the circle
    mask = distance_from_centre <= radius
    ##repeat the mask for each slice in the image set
    mask = np.repeat(mask[:, :, np.newaxis], x_dim, axis=2)
    #swap axes 0 and 2
    mask = np.swapaxes(mask, 0, 2)
    return mask

