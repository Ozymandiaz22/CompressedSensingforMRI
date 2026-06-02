import numpy as np
import os

samples_filepath = "C:\\Users\\osman\\Documents\\FYP Datasets\\Batch 5 Sample"
folders_filepath = "C:\\Users\\osman\\Documents\\FYP Datasets\\Batch 5 Results"
package_filepath = "C:\\Users\\osman\\Documents\\FYP Datasets\\Batch 5 package"
repackaged_filepath = "C:\\Users\\osman\\Documents\\FYP Datasets\\Batch 5 repackaged"

##each folder in the folders filepath is 1 run of reconstruction, with 1 sampling pattern
##the sampling patterns are in the samples filepath
## for each folder in folders filepath, grab the nii file, rename it according to a sampling pattern and a dictonaru of the code, along with the scan id (first 4 characters of the folder its in )

# create a dictionary to map the sampling pattern to the code
sampling_patterns = {
	"50_radial": {"percent_sampling": 50_0, "sampling_method": "radial"},
	"60_radial": {"percent_sampling": 60_0, "sampling_method": "radial"},
	"exLUT_radial_trajectory_y150_z150_pct69.6_a16.95_r1": {"percent_sampling": 69_6, "sampling_method": "radial"},
	"exLUT_radial_trajectory_y150_z150_pct79.8_a16.95_r1": {"percent_sampling": 79_8, "sampling_method": "radial"},
	"exLUT_radial_trajectory_y150_z150_pct90.2_a16.95_r1": {"percent_sampling": 90_2, "sampling_method": "radial"},
	"nrLUT_2D_Gauss_R1.11_pct90.0": {"percent_sampling": 90_0, "sampling_method": "2D Gauss"},
	"nrLUT_2D_Gauss_R1.25_pct80.0": {"percent_sampling": 80_0, "sampling_method": "2D Gauss"},
	"nrLUT_2D_Gauss_R1.43_pct70.0": {"percent_sampling": 70_0, "sampling_method": "2D Gauss"},
	"nrLUT_2D_Gauss_R1.67_pct60.0": {"percent_sampling": 60_0, "sampling_method": "2D Gauss"},
	"nrLUT_2D_Gauss_R2.00_pct50.0": {"percent_sampling": 50_0, "sampling_method": "2D Gauss"},
	"nrLUT_3D_Poisson_R1.11_M150x150E_perc90.1": {"percent_sampling": 90_1, "sampling_method": "3D Poisson"},
	"nrLUT_3D_Poisson_R1.25_M150x150E_perc80.0": {"percent_sampling": 80_0, "sampling_method": "3D Poisson"},
	"nrLUT_3D_Poisson_R1.43_M150x150E_perc69.9": {"percent_sampling": 69_9, "sampling_method": "3D Poisson"},
	"nrLUT_3D_Poisson_R1.67_M150x150E_perc59.9": {"percent_sampling": 59_9, "sampling_method": "3D Poisson"},
	"nrLUT_3D_Poisson_R2.00_M150x150E_perc50.0": {"percent_sampling": 50_0, "sampling_method": "3D Poisson"},
}

##for each nii file in the package filepath
#get a list of nii files
nii_files = [f for f in os.listdir(package_filepath) if f.endswith(".nii")]
print(nii_files)
#in each nii file, inf the percent smapling and sampling method from the filename
oldvnewname_dict = {}
for nii_file in nii_files:
    for pattern in sampling_patterns.keys():
        if pattern in nii_file:
            percent_sampling = sampling_patterns[pattern]["percent_sampling"]
            sampling_method = sampling_patterns[pattern]["sampling_method"]
            scan_id = nii_file[:4]
            new_filename = f"{scan_id}_{sampling_method}_{percent_sampling}.nii"
            oldvnewname_dict[nii_file] = new_filename

# print the old and new filenames
for old_name, new_name in oldvnewname_dict.items():
    print(f"{old_name} --> {new_name}")

##copy the nii files from the package filepath to the repackaged filepath with the new names
##save a copy of the nii files into repackaged filepath wiTH THE OLD NAMES
import shutil
os.makedirs(repackaged_filepath, exist_ok=True)
for old_name, new_name in oldvnewname_dict.items():
    old_path = os.path.join(package_filepath, old_name)
    print(f"Copying {old_path} to {repackaged_filepath} with new name {new_name}")
    new_path = os.path.join(repackaged_filepath, new_name)

    if os.path.exists(old_path):
        shutil.copy(old_path, new_path)
    else:
        print(f"Warning: File not found - {old_path}")