import sys

#input list of parametes from command line, if no parameters given, use default parameters from Default_Setup_Parameters.py
if len(sys.argv) > 1:
    folderpath = sys.argv[1]
    target_size = int(sys.argv[2])
    Wavelet_3D = sys.argv[3]
    Wavelet_3D_level = int(sys.argv[4])
    Percent_sampled = float(sys.argv[5])
    Regularization_parameter = float(sys.argv[6])
    Iteration_number = int(sys.argv[7])
    Tolerance = float(sys.argv[8])
else:
    import Default_Setup_Parameters
    folderpath = Default_Setup_Parameters.folderpath
    target_size = Default_Setup_Parameters.target_size
    Wavelet_3D = Default_Setup_Parameters.Wavelet_3D
    Wavelet_3D_level = Default_Setup_Parameters.Wavelet_3D_level
    Percent_sampled = Default_Setup_Parameters.Percent_sampled
    Regularization_parameter = Default_Setup_Parameters.Regularization_parameter
    Iteration_number = Default_Setup_Parameters.Iteration_number
    Tolerance = Default_Setup_Parameters.Tolerance

##Pass these parameters to the Live_parameters.py file, which will be imported into the reconstruction script to be used in the reconstruction process
