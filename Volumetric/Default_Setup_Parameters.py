### This file setus up parameters for the volumetric reconstruction, so that they can be changed easily though a command line interface when running the reconstruction script, without having to change the code of the reconstruction itself. This is done by importing this file into the reconstruction script and using the parameters defined here.

folderpath = "Volumetric/knee_mri_clinical_seq_batch2/1FB_1001820591____1FB,_3331562518/study_2f43b031/MR4_53c76c27"
target_size = 256
Wavelet_3D = 'haar'
Wavelet_3D_level = 5
Percent_sampled = 0.75
Regularization_parameter = 0.01
Iteration_number = 500
Tolerance = 1e-6