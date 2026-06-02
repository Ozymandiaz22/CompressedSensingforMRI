import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sigpy as sp
from scipy import stats
import pylops
import pylops.optimization.sparsity 
from PIL import Image
import pydicom as dicom
import os
import nibabel as nib


import radiaslsampling as rss
# print(sys.argv)
if len(sys.argv) > 1:
    folderpath = sys.argv[1]
    # print("Folder path:", folderpath)
    target_size = int(sys.argv[2])
    # print("target_size:", target_size)
    Wavelet_3D = sys.argv[3]
    # print("Wavelet_3D:", Wavelet_3D)
    Wavelet_3D_level = int(sys.argv[4])
    # print("Wavelet_3D_level:", Wavelet_3D_level)
    Regularization_Parameter = float(sys.argv[5])
    # print("Regularization_Parameter:", Regularization_Parameter)
    Iteration_number = int(sys.argv[6])
    # print("Iteration_number:", Iteration_number)
    Tolerance = float(sys.argv[7])
    # print("Tolerance:", Tolerance)
    Output_File = sys.argv[8]
    # print("Output_File:", Output_File)
    mask_filepath = sys.argv[9]
    # print("mask_filepath:", mask_filepath)
    nifpath = sys.argv[10]
    # print("nifpath:", nifpath)
else:
    print("Using default parameters. To specify parameters, run the script with the following arguments:")
    print("python Volumetric_kspace_maskfrombitmap.py <folderpath> <target_size> <Wavelet_3D> <Wavelet_3D_level> <Percent_sampled> <Iteration_number> <Regularization_Parameter> <Tolerance> <Output_File> <mask_filename>")
    print("Example:")
    print("python Volumetric_kspace_maskfrombitmap.py ./dicom_images 128 db6 1 0.2 100 0.01 1e-6 ./output/mask_from_bitmap ./mask.bmp")
    folderpath = "C:\\Users\\osman\\Documents\\FYP Datasets\\Batch 1\\HV15_MRD\\5927"
    target_size = 128
    Wavelet_3D = 'haar'
    Wavelet_3D_level = 5
    Percent_sampled = 0.75
    Regularization_parameter = 0.1
    Iteration_number = 100
    Tolerance = 1e-6
    Output_File = "C:\\Users\\osman\\Documents\\GitHub\\CompressedSensingforMRI\\Batch1 tests"
    mask_filepath = "C:\\Users\\osman\\Documents\\GitHub\\CompressedSensingforMRI\\mask.bmp"

# print("Wavelet_3D:", Wavelet_3D)

plt.close('all')
#this file is to do volumetric reconstruction with wavelet sparsity regularization
files = [str(folderpath + "/" + f  ) for f in os.listdir(folderpath) if f.endswith('.MRD')]

##read mrd file using evom3dread_converted
import evom3dread_converted as evom3d
mrd = evom3d.mread(files[0])
data = mrd['data']
# print("data shape:", data.shape)   
#extract k spce images from the data
kspace1 = data[:, :, :, 0, 0, 0]
kspace2 = data[:, :, :, 0, 0, 1]
#this should be 150x150x150 for all real data sets
nl, ny, nx = kspace1.shape
# print("kspace shape:", kspace1.shape)

Fop_3D = pylops.signalprocessing.FFTND(dims=(nl, ny, nx),ifftshift_before=True, dtype=np.complex128,fftshift_after=False)
Wop3D = pylops.signalprocessing.DWTND(dims=(target_size, target_size, target_size), wavelet=Wavelet_3D, level=Wavelet_3D_level, axes=(0, 1, 2), dtype=np.complex128)


#set up the basis pursuit problem to be solved with FISTA algorithm assuming the image is sparse in wavelet domain
#our image comes to us undersampled in the fourier domain, so we need to use the fourier operator as our forward model
#we can use the wavelet transform as our sparsifying transform, and then use the inverse wavelet transform to reconstruct the image from the wavelet coefficients





#samples selected / load sampling mask
# If a bitmap filepath is provided (mask_filepath), use it. Otherwise use generated samples list.

##open the mask bitmap and convert to a boolean array
mask_image = Image.open(mask_filepath).convert('L')  # Convert to grayscale
mask_array = np.array(mask_image)
sampling_mask = mask_array  > 128
# Ensure numpy does not truncate large array prints so the full sampling mask is shown
##if mask is not the same size as the k space, zero pad it to match the k space dimensions

##stack the 2D mask to create a 3D mask with the same sampling pattern for each slice
sampling_mask = np.stack([sampling_mask] * nl, axis=0)
# Extract the indices of the sampled points from the 3D mask
samples = np.where(sampling_mask.flatten())[0]


# show the first slice of the sampling mask (samples=1, missing=0)
# plt.imshow(sampling_mask[0], cmap='gray', origin='lower')
# plt.title('Sampling Mask for First Slice')
# plt.imsave(f"{Output_File}/sampling_mask.png", sampling_mask[0], cmap='gray')
# plt.close()

#restriction operator selects samples from the fourier domain
Rop = pylops.Restriction(nl * ny * nx, samples, axis=-1, dtype=np.complex128)
#our sparcifying tarnsform is the 3D wavelet transform
Sop = Wop3D.H

##define the pad operator to pad the images to the target size
padl = (target_size - nx) / 2
pad_lengths = ((int(padl), int(padl)), (int(padl), int(padl)), (int(padl), int(padl)))
Pad_Op = pylops.Pad(dims=(nl, ny, nx), pad=pad_lengths, dtype=np.complex128)


#we will seek to solve the analysis problem: given as
#argmin||y - Op x||_2^2 +epsilon*||SOp^H x||_1
#Sop is the wavelet transform, and Sop^H is the inverse wavelet transform
#Op is the forward model, which is the restriction operator composed with the fourier operator
#our undersampled k space is our measurements y, and our variable x is the image we want to reconstruct
# print("Setting up the forward operator and measurements...")
# ##operator sizes
# print("Fop_3D shape:", Fop_3D.shape)
# print("Pad_Op shape:", Pad_Op.shape)
# print("Rop shape:", Rop.shape)
# print("Sop shape:", Sop.shape)



Op = Rop * Fop_3D @ Pad_Op.H 
#forward operator generated
y1 = Rop * np.asarray(kspace1).ravel('K')
y2 = Rop * np.asarray(kspace2).ravel('K')
#measuremnents in the fourier domain generated in (nl*nx*samples,) shape

#we can now use the FISTA algorithm to solve the optimization problem
epsilon = Regularization_Parameter

#define the initial guess as the inverse transform of the masked k space data
masked_kspace_1 = kspace1 * sampling_mask
x0_1 = Fop_3D.H * masked_kspace_1.ravel('K')
x0_1 = Pad_Op * x0_1
x0_1 = x0_1.ravel('K')

masked_kspace_2 = kspace2 * sampling_mask
x0_2 = Fop_3D.H * masked_kspace_2.ravel('K')
x0_2 = Pad_Op * x0_2
x0_2 = x0_2.ravel('K')


#we will solve the problem for each image in the stack
recons1 = []
recons2 = []
#print shapes of all the variables


#images = kspace * Fop_3D.H * (1/(nx*ny*nl))
images = Fop_3D.H * kspace1.ravel('K')
images = images.reshape((nl, ny, nx))
#images = np.swapaxes(images, 0, 1)



# for i in range(len(y)):
#     print("reconstructing image", i)
#     (x, niter, cost) = pylops.optimization.sparsity.fista(Op, y[i], eps=epsilon, x0=x0, niter=100, SOp=Sop, tol=1e-6)
#     recons.append(x.reshape((ny, nx)))

#reconstruct the whole stack at once
(x1, niter1, cost1) = pylops.optimization.sparsity.fista(Op, y1, eps=epsilon, x0=x0_1, niter=Iteration_number, SOp=Sop,show=False,alpha=0.2, threshkind='soft')
(x2, niter2, cost2) = pylops.optimization.sparsity.fista(Op, y2, eps=epsilon, x0=x0_2, niter=Iteration_number, SOp=Sop,show=False,alpha=0.2, threshkind='soft')
#un ravel the reconstructed stack
# print("x shape:", x1.shape)
recons1 = Pad_Op.H * x1.reshape((target_size, target_size, target_size))
# print("recons shape:", recons1.shape)
# print("x shape:", x2.shape)
recons2 = Pad_Op.H * x2.reshape((target_size, target_size, target_size))
# print("recons shape:", recons2.shape)

# print("cost1:", cost1)
# print("cost2:", cost2)
##generate original images from the k space data by applying the inverse fourier transform to each slice in the stack
##3d ifft

# Process both reconstructions
for recon_name, recons_data, x_data, cost_data in [('1', recons1, x1, cost1), ('2', recons2, x2, cost2)]:
    difference_images = []
    normalised_difference_images = []

    # Vectorized computation of difference images
    abs_recons = np.abs(recons_data)
    abs_images = np.abs(images)
    peak_recons = np.maximum(np.amax(abs_recons, axis=(1, 2), keepdims=True), 1.0)
    peak_images = np.maximum(np.amax(abs_images, axis=(1, 2), keepdims=True), 1.0)

    for i in range(len(recons_data)):
        difference_images.append(abs_recons[i] - abs_images[i])
        normalised_difference_images.append((abs_recons[i] / peak_recons[i]) - (abs_images[i] / peak_images[i]))

    ####masked k spaces
    masked_kspace = kspace1 * (sampling_mask)

    ##sampling mask name
    mask_name = os.path.basename(mask_filepath).split('.')[0]
    data_name = os.path.basename(files[0]).split('.')[0]

    ##generate a nifti image of the reconstructed stack
    nifti_img = nib.Nifti1Image(abs(recons_data), affine=None)
    nib.save(nifti_img, f"{nifpath}/{recon_name}_image_{data_name}_{Wavelet_3D}_Reg={epsilon}_Data#{recon_name}_SamplingMask={mask_name}.nii")

    #Output file is now assumed to exist as it is being passed as an argument
    # Precompute normalized arrays and log transforms for faster loop execution

    abs_recons_all = np.abs(recons_data)
    abs_images_all = np.abs(images)
    abs_normalised_diff = np.abs(normalised_difference_images)
    kspace_log = np.log(np.abs(kspace1) + 1e-10)
    masked_kspace_log = np.log(np.abs(masked_kspace) + 1e-10)

    # Save individual images using PIL for 5-10x faster I/O
    #set max pixel value

    cmap = plt.get_cmap('gray')  # Choose a colormap for the difference images
    for i in range(len(recons_data)):
        # Convert to uint8 for PIL (normalized to 0-255)
        #max is the max of the original
        i_max = np.amax(np.abs(images[i]))
        diffmax = np.amax(np.abs(difference_images[i]))

        colour_diff= cmap((np.abs(difference_images[i]) / i_max))  # Normalize for colormap
        recon_image = (np.abs(recons_data[i]) / i_max * 255).astype(np.uint8)
        original_image = (np.abs(images[i]) / i_max * 255).astype(np.uint8)

        # Image.fromarray(recon_image).save(f"{Output_File}/reconstructed_{recon_name}_{i}.png")
        # Image.fromarray(original_image).save(f"{Output_File}/original_{recon_name}_{i}.png")
        # Image.fromarray((colour_diff[:,:,:3]*255).astype(np.uint8)).save(f"{Output_File}/difference_{recon_name}_{i}.png")##

    ##save slice 69 of the sampling mask and masked k space
    # Image.fromarray((sampling_mask[69] * 255).astype(np.uint8)).save(f"{Output_File}/sampling_mask_slice_69.png")
    # masked_kspace_69_log = np.log1p(np.abs(masked_kspace[69]))
    # Image.fromarray((masked_kspace_69_log / np.amax(masked_kspace_69_log) * 255).astype(np.uint8)).save(f"{Output_File}/masked_kspace_slice_69.png")
    # # Save the log of the k-space for slice 69
    # kspace_69_log = np.log1p(np.abs(kspace1[69]))
    # Image.fromarray((kspace_69_log / np.amax(kspace_69_log) * 255).astype(np.uint8)).save(f"{Output_File}/kspace_slice_69.png")
    # Save comparison figures only for key slices (every 10th slice or first/middle/last)
    comparison_slices = [0, len(recons_data)//2, len(recons_data)-1] + list(range(0, len(recons_data), 10))
    comparison_slices = sorted(set(comparison_slices))  # Remove duplicates and sort

    for i in comparison_slices:
        if i >= len(recons_data):
            continue
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        axs[0, 0].imshow(abs_images_all[i], cmap='gray')
        axs[0, 0].set_title('Original Image')
        axs[0, 0].axis('off')
        axs[0, 1].imshow(abs_recons_all[i], cmap='gray')
        axs[0, 1].set_title('Reconstructed Image')
        axs[0, 1].axis('off')
        axs[0, 2].imshow(abs_normalised_diff[i], cmap='gray')
        axs[0, 2].set_title('Difference Image')
        axs[0, 2].axis('off')
        axs[1, 0].imshow(kspace_log[i], cmap='gray')
        axs[1, 0].set_title('K-Space')
        axs[1, 0].axis('off')
        axs[1, 1].imshow(np.abs(sampling_mask[i]), cmap='gray')
        axs[1, 1].set_title('Sampling Mask')
        axs[1, 1].axis('off')
        axs[1, 2].imshow(masked_kspace_log[i], cmap='gray')
        axs[1, 2].set_title('Masked K-Space')
        axs[1, 2].axis('off')
        plt.savefig(f"{Output_File}/comparison_{recon_name}_{i}_Data#{recon_name}.png", dpi=100, bbox_inches='tight')
        plt.close(fig)

    ##plot the cost function over iterations
    plt.figure()
    plt.plot(cost_data)
    plt.title(f'Cost Function over Iterations ({recon_name})')
    plt.xlabel('Iteration')
    plt.yscale('log')
    plt.ylabel('Cost')
    plt.savefig(f"{Output_File}/cost_function_{recon_name}.png")
    plt.close()

    #save the cost function values to a text file
    with open(f"{Output_File}/cost_values_{recon_name}.txt", 'w') as f:
        for cost in cost_data:
            f.write(f"{cost}\n")

    ##save a text file with the final data fidelity and regularization terms
    final_data_fidelity = np.linalg.norm(Op * x_data - y1)**2
    final_regularization = np.linalg.norm(Sop.H * x_data, 1)
    with open(f"{Output_File}/final_terms_{recon_name}.txt", 'w') as f:
        f.write(f"Final Data Fidelity: {final_data_fidelity}\n")
        f.write(f"Final Regularization: {final_regularization}\n")